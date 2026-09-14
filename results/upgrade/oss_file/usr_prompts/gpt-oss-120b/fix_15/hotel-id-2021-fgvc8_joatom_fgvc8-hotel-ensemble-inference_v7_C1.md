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

0.612006398922285

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'The update removes the failing model‑loading step and adds a safe fallback that builds a simple submission using the five most common hotel IDs from the training metadata. It first checks whether any of the specified model files exist; if none are found, it computes the top‑5 frequent `hotel_id`s and assigns that list to every test image, guaranteeing a correctly formatted `submission.csv` without runtime errors. This fixes the FileNotFoundError and subsequent undefined‑variable errors while still preserving the original structure for potential future model use.'
- What this solution (achieved 3e-05) has done: 'The update caches the test DataLoader so images are read only once and reused across all models, adds CuDNN benchmarking for faster GPU kernels, and streamlines the fallback distance‑to‑centroid loop with vectorized string construction while keeping all original logic intact. These changes eliminate repeated I/O and reduce Python‑level overhead, allowing the script to finish well within the 600‑second limit without altering any model architecture or prediction semantics.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from fastai.vision.all import *

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)  # limit CPU threads for predictable performance

models = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",  # v7
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",  # v8
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",  # v11
    "../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",  # kaggle v2
]

submission_path = "../input/hotel-id-2021-hotel-id-2021-fgvc8/sample_submission.csv"
submission = pd.read_csv(submission_path)
test = submission.copy()
test["image"] = "../input/hotel-id-2021-fgvc8/test_images/" + test["image"]

probs = None
any_model_loaded = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    bs = 512
    num_dl_workers = min(8, os.cpu_count() or 0)

    test_dl = None  # will be created once and reused
    for model_path in models:
        model_file = Path(model_path)
        if model_file.exists():
            try:
                learn = load_learner(
                    fname=model_file, cpu=False, pickle_module=__import__("dill")
                )
                if test_dl is None:
                    test_dl = learn.dls.test_dl(
                        test["image"], bs=bs, num_workers=num_dl_workers
                    )
                probs_temp, _ = learn.tta(dl=test_dl, n=5)
                probs = probs_temp if probs is None else probs + probs_temp
                any_model_loaded = True
            except Exception as e:
                print(f"Warning: failed to load or run model {model_file}: {e}")

if not any_model_loaded:
    train_path = "../input/hotel-id-2021-fgvc8/train.csv"
    train_images_root = Path("../input/hotel-id-2021-fgvc8/train_images")
    train_df = pd.read_csv(train_path)

    train_df["img_path"] = (
        train_images_root / train_df["chain"].astype(str) / train_df["image"]
    ).astype(str)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader("img_path"),
        get_y=ColReader("hotel_id"),
        splitter=FuncSplitter(lambda o: False),  # no validation split
        item_tfms=Resize(256),
    )
    dls = dblock.dataloaders(
        train_df,
        bs=bs,
        num_workers=min(8, os.cpu_count() or 0),
        shuffle=False,
    )

    learn = vision_learner(dls, resnet34, pretrained=True, metrics=[])
    learn.model.to(device)
    learn.model.eval()

    embeddings = []
    hotel_ids = []
    with torch.no_grad():
        for xb, yb in dls.train:
            xb = xb.to(device)
            emb = learn.model(xb)
            embeddings.append(emb.cpu())
            hotel_ids.extend(yb.cpu().numpy().tolist())

    train_emb = torch.cat(embeddings)  # (N_train, D)
    train_hotel = np.array(hotel_ids)  # (N_train,)

    centroids = {}
    for hid in np.unique(train_hotel):
        mask = train_hotel == hid
        centroids[int(hid)] = train_emb[mask].mean(dim=0)

    if test_dl is None:
        test_dl = learn.dls.test_dl(
            test["image"], bs=bs, num_workers=min(8, os.cpu_count() or 0)
        )

    centroid_ids = list(centroids.keys())
    centroid_tensor = torch.stack([centroids[hid] for hid in centroid_ids]).to(device)

    pred_lists = []
    with torch.no_grad():
        for xb in test_dl:
            xb = xb[0] if isinstance(xb, (list, tuple)) else xb
            xb = xb.to(device)
            emb = learn.model(xb)  # (batch, D)
            dists = torch.cdist(emb, centroid_tensor)  # (batch, n_hotels)
            topk = torch.topk(-dists, k=5, dim=1)  # nearest = largest -dist
            top_idx = topk.indices.cpu().numpy()
            pred_lists.extend(
                [" ".join(str(centroid_ids[i]) for i in row) for row in top_idx]
            )

    preds = pd.Series(pred_lists, index=submission.index)

else:
    preds_idx = probs.topk(5, dim=1)[1]  # indices of top‑5 classes per sample
    vocab = learn.dls.vocab
    preds_list = []
    for row in preds_idx:
        pred_ids = [str(vocab[idx]) for idx in row]
        preds_list.append(" ".join(pred_ids))
    preds = pd.Series(preds_list, index=submission.index)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2921978.py in <cell line: 0>()
     17 
     18 submission_path = "../input/hotel-id-2021-hotel-id-2021-fgvc8/sample_submission.csv"
---> 19 submission = pd.read_csv(submission_path)
     20 test = submission.copy()
     21 test["image"] = "../input/hotel-id-2021-fgvc8/test_images/" + test["image"]

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/hotel-id-2021-hotel-id-2021-fgvc8/sample_submission.csv'

## === cell 1
submission["hotel_id"] = preds
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
submission.head()

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4047414613.py in <cell line: 0>()
----> 1 submission["hotel_id"] = preds
      2 output_path = "submission.csv"
      3 submission.to_csv(output_path, index=False)
      4 print(f"Submission file written to {output_path}")
      5 submission.head()

NameError: name 'preds' is not defined
