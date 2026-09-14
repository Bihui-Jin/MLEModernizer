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

0.5955733771154317

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import fastai
from fastai.vision.all import *

import dill



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
models = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",
    "../input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",
]



## === cell 3
BASE = Path("/kaggle")
INPUT = BASE / "input" / "hotel-id-2021-fgvc8"
WORKING = BASE / "working"

assert INPUT.exists(), f"Expected competition data at {INPUT} but it does not exist."

train_csv_path = INPUT / "train.csv"
sample_sub_path = INPUT / "sample_submission.csv"
train_img_root = INPUT / "train_images"
test_img_root = INPUT / "test_images"

assert train_csv_path.exists()
assert sample_sub_path.exists()
assert train_img_root.exists()
assert test_img_root.exists()

train_df = pd.read_csv(
    train_csv_path,
    dtype={
        "image": "string",
        "chain": "int32",
        "hotel_id": "string",
        "timestamp": "int64",
    },
)
submission = pd.read_csv(
    sample_sub_path, dtype={"image": "string", "hotel_id": "string"}
)

train_df.head(), submission.head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
parsers.pyx in pandas._libs.parsers.TextReader._convert_tokens()

TypeError: Cannot cast array data from dtype('O') to dtype('int64') according to the rule 'safe'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/346858135.py in <cell line: 0>()
     16 
     17 # Speed: specify dtypes to reduce CSV parsing overhead; values are used as strings anyway.
---> 18 train_df = pd.read_csv(
     19     train_csv_path,
     20     dtype={

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    624 
    625     with parser:
--> 626         return parser.read(nrows)
    627 
    628 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read(self, nrows)
   1921                     columns,
   1922                     col_dict,
-> 1923                 ) = self._engine.read(  # type: ignore[attr-defined]
   1924                     nrows
   1925                 )

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/c_parser_wrapper.py in read(self, nrows)
    232         try:
    233             if self.low_memory:
--> 234                 chunks = self._reader.read_low_memory(nrows)
    235                 # destructive to chunks
    236                 data = _concatenate_chunks(chunks)

parsers.pyx in pandas._libs.parsers.TextReader.read_low_memory()

parsers.pyx in pandas._libs.parsers.TextReader._read_rows()

parsers.pyx in pandas._libs.parsers.TextReader._convert_column_data()

parsers.pyx in pandas._libs.parsers.TextReader._convert_tokens()

ValueError: invalid literal for int() with base 10: '2018-04-16 17:01:49'

## === cell 4
test = submission.copy()
test["image_path"] = (
    test_img_root.as_posix() + "/" + test["image"].astype(str)
).astype(str)
test.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2653183782.py in <cell line: 0>()
      1 # Speed: avoid per-row Python os.path.exists checks on test (very slow); dataset should be consistent.
----> 2 test = submission.copy()
      3 test["image_path"] = (
      4     test_img_root.as_posix() + "/" + test["image"].astype(str)
      5 ).astype(str)

NameError: name 'submission' is not defined

## === cell 5
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed, reproducible=True)

torch.backends.cudnn.benchmark = True



## === cell 6
existing_model_files = [Path(m) for m in models if Path(m).exists()]
existing_model_files



## === cell 7
probs = None
learn = None
vocab = None
test_dl = None

if len(existing_model_files) > 0:
    test_items = test["image_path"].tolist()

    ncpu = os.cpu_count() or 2
    num_workers = min(4, max(2, ncpu // 2))

    first_model_path = existing_model_files[0]
    learn = load_learner(fname=Path(first_model_path), cpu=False, pickle_module=dill)
    vocab = np.asarray(learn.dls.vocab, dtype=object)

    try:
        base_bs = int(getattr(learn.dls, "bs", 32))
    except Exception:
        base_bs = 32
    test_bs = max(base_bs, 128)

    test_dl = learn.dls.test_dl(
        test_items,
        with_labels=False,
        bs=test_bs,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    learn.model.eval()
    with torch.inference_mode():
        probs_temp, _ = learn.get_preds(dl=test_dl, reorder=False)
    probs = probs_temp.float()
    del probs_temp
    del learn

    for model_path in existing_model_files[1:]:
        learn = load_learner(fname=Path(model_path), cpu=False, pickle_module=dill)
        learn.model.eval()
        with torch.inference_mode():
            probs_temp, _ = learn.get_preds(dl=test_dl, reorder=False)
        probs.add_(probs_temp.float())
        del probs_temp
        del learn

    probs.div_(len(existing_model_files))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

else:
    train_df = train_df.copy()

    train_df["chain"] = train_df["chain"].astype(str)
    train_df["image"] = train_df["image"].astype(str)
    train_df["image_path"] = (
        train_img_root.as_posix() + "/" + train_df["chain"] + "/" + train_df["image"]
    )

    paths = train_df["image_path"].to_numpy()
    exists_mask = np.fromiter(
        (os.path.exists(p) for p in paths), dtype=bool, count=len(paths)
    )
    train_df = train_df.loc[exists_mask].reset_index(drop=True)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader("image_path"),
        get_y=ColReader("hotel_id"),
        splitter=RandomSplitter(valid_pct=0.1, seed=seed),
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
    )
    dls = dblock.dataloaders(train_df, bs=32)

    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(1)

    vocab = np.asarray(learn.dls.vocab, dtype=object)

    ncpu = os.cpu_count() or 2
    num_workers = min(4, max(2, ncpu // 2))
    test_bs = 128

    test_dl = learn.dls.test_dl(
        test["image_path"].tolist(),
        with_labels=False,
        bs=test_bs,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    learn.model.eval()
    with torch.inference_mode():
        probs, _ = learn.get_preds(dl=test_dl, reorder=False)

probs.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1203080395.py in <cell line: 0>()
     59 
     60 else:
---> 61     train_df = train_df.copy()
     62 
     63     # Keep core logic for fallback training path; optimize existence check and dataloader creation overhead slightly.

NameError: name 'train_df' is not defined

## === cell 8
topk_idx = probs.topk(5, dim=1).indices.cpu().numpy()
topk_labels = np.take(vocab, topk_idx)

preds = [" ".join(row.tolist()) for row in topk_labels]

preds[:5], len(preds)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/298802463.py in <cell line: 0>()
      1 # Speed: use torch topk then map to labels with numpy take for efficient indexing.
----> 2 topk_idx = probs.topk(5, dim=1).indices.cpu().numpy()
      3 topk_labels = np.take(vocab, topk_idx)
      4 
      5 # Speed: faster join via list comprehension over already-materialized small rows (5 elems each).

AttributeError: 'NoneType' object has no attribute 'topk'

## === cell 9
submission_out = submission.copy()
submission_out["hotel_id"] = preds

out_path = WORKING / "submission.csv"
submission_out.to_csv(out_path, index=False)

out_path, submission_out.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/268842408.py in <cell line: 0>()
----> 1 submission_out = submission.copy()
      2 submission_out["hotel_id"] = preds
      3 
      4 out_path = WORKING / "submission.csv"
      5 submission_out.to_csv(out_path, index=False)

NameError: name 'submission' is not defined
