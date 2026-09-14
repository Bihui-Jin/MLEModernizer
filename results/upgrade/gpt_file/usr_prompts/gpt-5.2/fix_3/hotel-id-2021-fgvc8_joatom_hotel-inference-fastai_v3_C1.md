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

- What this solution (achieved 0.00209) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when the external pretrained FastAI `.pkl` is missing in this environment (the root cause of your crash). Concretely, I add a safe model-loading block that tries the provided path and, on failure, falls back to a deterministic “most frequent hotels” baseline derived from `train.csv`, which is score-positive vs. random and guarantees end-to-end execution. I also correct the cell numbering and make the test image paths robust to the actual Kaggle directory layout. The rest of your inference logic (top-5 formatting, submission schema) is preserved.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import pandas as pd
import numpy as np

import fastai
from fastai.vision.all import *
import dill

import torch
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
sub_path_candidates = [
    Path("../input/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("/kaggle/data/hotel-id-2021-fgvc8/sample_submission.csv"),
]
sub_path = next((p for p in sub_path_candidates if p.exists()), None)
if sub_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in expected locations: {sub_path_candidates}"
    )

submission = pd.read_csv(sub_path)
test = submission.copy()

test_img_dir_candidates = [
    sub_path.parent / "test_images",
    Path("../input/hotel-id-2021-fgvc8/test_images"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/test_images"),
    Path("/kaggle/data/hotel-id-2021-fgvc8/test_images"),
]
test_img_dir = next((p for p in test_img_dir_candidates if p.exists()), None)
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images dir in expected locations: {test_img_dir_candidates}"
    )

test["image"] = test["image"].apply(lambda x: str(test_img_dir / x))
test.head()



## === cell 3
learn = None
learner_path_candidates = [
    Path("../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"),
    Path(
        "/kaggle/input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl"
    ),
]

for p in learner_path_candidates:
    if p.exists():
        learn = load_learner(fname=p, cpu=False, pickle_module=dill)
        break

learn is not None, learner_path_candidates



## === cell 4
if learn is not None:
    test_dl = learn.dls.test_dl(test)
    probs, _ = learn.tta(dl=test_dl)
    preds_idx = probs.topk(5)[1]
    preds = [" ".join(map(str, learn.dls.vocab[pred])) for pred in preds_idx]
else:
    train_csv_candidates = [
        Path("../input/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/train.csv"),
    ]
    train_csv = next((p for p in train_csv_candidates if p.exists()), None)
    if train_csv is None:
        raise FileNotFoundError(
            f"Could not find train.csv in expected locations: {train_csv_candidates}"
        )

    train_img_dir_candidates = [
        train_csv.parent / "train_images",
        Path("../input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/train_images"),
    ]
    train_img_dir = next((p for p in train_img_dir_candidates if p.exists()), None)
    if train_img_dir is None:
        raise FileNotFoundError(
            f"Could not find train_images dir in expected locations: {train_img_dir_candidates}"
        )

    train_df = pd.read_csv(train_csv, usecols=["image", "chain", "hotel_id"])
    train_df["image_path"] = train_df.apply(
        lambda r: str(train_img_dir / str(r["chain"]) / r["image"]), axis=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = create_body(resnet50, pretrained=True, cut=-2).to(device).eval()

    item_tfms = Resize(224, method="squish")
    batch_tfms = [IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
    dblock = DataBlock(
        blocks=(ImageBlock,),
        get_x=ColReader("image_path"),
        splitter=FuncSplitter(lambda o: True),  # dummy
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    train_dl = dblock.dataloaders(train_df, bs=64, num_workers=2).train
    test_df = pd.DataFrame({"image_path": test["image"].values})
    test_dl = dblock.dataloaders(test_df, bs=64, num_workers=2).train

    @torch.no_grad()
    def encode_dl(dl):
        feats = []
        for batch in dl:
            x = batch[0].to(device)
            f = backbone(x)
            if f.ndim == 4:
                f = f.mean(dim=(2, 3))
            f = F.normalize(f, p=2, dim=1)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    train_emb = encode_dl(train_dl)  # (N_train, D)
    test_emb = encode_dl(test_dl)  # (N_test, D)

    train_hotel_ids = train_df["hotel_id"].astype(str).values

    def topk_unique_hotels(scores_row, k=5):
        idxs = torch.topk(scores_row, k=min(50, scores_row.numel())).indices.numpy()
        out = []
        seen = set()
        for idx in idxs:
            hid = train_hotel_ids[idx]
            if hid not in seen:
                out.append(hid)
                seen.add(hid)
                if len(out) == k:
                    break
        if len(out) < k:
            top5 = (
                pd.Series(train_hotel_ids)
                .value_counts()
                .head(k)
                .index.astype(str)
                .tolist()
            )
            for hid in top5:
                if hid not in seen:
                    out.append(hid)
                    seen.add(hid)
                if len(out) == k:
                    break
        return " ".join(out[:k])

    preds = []
    chunk = 512
    train_emb_t = train_emb.t()  # (D, N_train)
    for i in range(0, test_emb.shape[0], chunk):
        te = test_emb[i : i + chunk]  # (b, D)
        sims = te @ train_emb_t  # (b, N_train)
        for r in range(sims.shape[0]):
            preds.append(topk_unique_hotels(sims[r], k=5))

preds[:3], len(preds), len(submission)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/834637116.py in <cell line: 0>()
     46     # Pretrained backbone; use fastai's built-in resnet50 (available with fastai)
     47     # We take features right before the classification head.
---> 48     backbone = create_body(resnet50, pretrained=True, cut=-2).to(device).eval()
     49 
     50     # Minimal transform compatible with common ImageNet pretrained encoders

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in create_body(model, n_in, pretrained, cut)
     85         ll = list(enumerate(model.children()))
     86         cut = next(i for i,o in reversed(ll) if has_pool_type(o))
---> 87     return cut_model(model, cut)
     88 
     89 # %% ../../nbs/21_vision.learner.ipynb 20

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in cut_model(model, cut)
     74 def cut_model(model, cut):
     75     "Cut an instantiated model"
---> 76     if   isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
     77     elif callable(cut): return cut(model)
     78     raise NameError("cut must be either integer or a function")

AttributeError: 'function' object has no attribute 'children'

## === cell 5
submission["hotel_id"] = preds
out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

assert list(submission.columns) == ["image", "hotel_id"]
assert len(submission) == len(pd.read_csv(sub_path))
out_path, submission.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1560556944.py in <cell line: 0>()
----> 1 submission["hotel_id"] = preds
      2 out_path = Path("submission.csv")
      3 submission.to_csv(out_path, index=False)
      4 
      5 assert list(submission.columns) == ["image", "hotel_id"]

NameError: name 'preds' is not defined
