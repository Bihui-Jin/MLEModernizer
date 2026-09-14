# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
from pathlib import Path
import gc

import numpy as np
import pandas as pd

import fastai
from fastai.vision.all import *

import dill

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

set_seed(42, reproducible=True)



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
models = [
    "../input/fgvc8hotel/export_res50_HQAdam.pkl",  # v3
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",  # v7
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",  # v8
    "../input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",  # v5
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",  # v11
    "../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",  # kaggle v2
]



## === cell 3
BASE = Path("../input/hotel-id-2021-fgvc8")
if not BASE.exists():
    BASE = Path("../input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8")

TRAIN_CSV = BASE / "train.csv"
SAMPLE_SUB = BASE / "sample_submission.csv"
TRAIN_IMG_DIR = BASE / "train_images"
TEST_IMG_DIR = BASE / "test_images"

TRAIN_CSV.exists(), SAMPLE_SUB.exists(), TRAIN_IMG_DIR.exists(), TEST_IMG_DIR.exists()



## === cell 4
submission = pd.read_csv(SAMPLE_SUB)
test_fnames = [TEST_IMG_DIR / f for f in submission["image"].tolist()]

missing = [str(p) for p in test_fnames if not p.exists()]
len(test_fnames), len(missing), (missing[:3] if missing else None)



## === cell 5
existing_models = [Path(m) for m in models if Path(m).exists()]


def _tfms_signature(learn: Learner):
    dls = learn.dls

    def _names(tfms):
        return tuple(type(t).__name__ for t in (tfms or []))

    return (
        type(dls).__name__,
        getattr(dls, "bs", None),
        _names(getattr(dls, "after_item", None)),
        _names(getattr(dls, "before_batch", None)),
        _names(getattr(dls, "after_batch", None)),
    )


class _PILCachedItems:
    def __init__(self, paths, cache=None):
        self.paths = list(paths)
        self._cache = cache if cache is not None else {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        key = str(p)
        im = self._cache.get(key, None)
        if im is None:
            im = create_image(p)
            self._cache[key] = im
        return im


def _build_fast_test_dl(learn: Learner, items):
    cpu_cnt = os.cpu_count() or 2
    nw = min(6, max(2, cpu_cnt // 2))
    bs = getattr(learn.dls, "bs", None)

    with torch.no_grad():
        try:
            return learn.dls.test_dl(
                items,
                with_labels=False,
                shuffle=False,
                bs=bs,
                num_workers=nw,
                persistent_workers=(nw > 0),
                prefetch_factor=4,
                pin_memory=torch.cuda.is_available(),
            )
        except TypeError:
            try:
                return learn.dls.test_dl(
                    items,
                    with_labels=False,
                    shuffle=False,
                    bs=bs,
                    num_workers=nw,
                    persistent_workers=(nw > 0),
                )
            except TypeError:
                return learn.dls.test_dl(
                    items,
                    with_labels=False,
                    shuffle=False,
                    bs=bs,
                    num_workers=nw,
                )


_TENSOR_BATCH_CACHE = {}  # (tfms_sig, seed) -> list[Tensor] (cpu)


def _cached_tta_batches(test_dl, tfms_sig, seed: int):
    key = (tfms_sig, int(seed))
    cached = _TENSOR_BATCH_CACHE.get(key, None)
    if cached is not None:
        return cached

    set_seed(seed, reproducible=True)
    batches = []
    for batch in test_dl:
        xb = batch[0] if isinstance(batch, (tuple, list)) else batch
        batches.append(xb.detach().cpu())
    _TENSOR_BATCH_CACHE[key] = batches
    return batches


def _predict_probs_one_pass_fast_from_batches(learn: Learner, cpu_batches):
    model = learn.model
    model.eval()
    device = next(model.parameters()).device

    out_chunks = []
    softmax = torch.softmax
    with torch.inference_mode():
        for xb_cpu in cpu_batches:
            xb = xb_cpu.to(device, non_blocking=True)
            logits = model(xb)
            out_chunks.append(softmax(logits, dim=1).detach().cpu())
    return torch.cat(out_chunks, dim=0)


def _predict_probs_tta_fast_cached(
    learn: Learner, test_dl, tfms_sig, n_tta: int = 4, base_seed: int = 42
):
    probs_sum = None
    for i in range(n_tta):
        seed = base_seed + i
        cpu_batches = _cached_tta_batches(test_dl, tfms_sig=tfms_sig, seed=seed)
        preds = _predict_probs_one_pass_fast_from_batches(learn, cpu_batches)
        probs_sum = preds if probs_sum is None else (probs_sum + preds)
    return probs_sum


def _free_learner(_learn: Learner):
    try:
        _learn.dls = None
    except Exception:
        pass
    try:
        _learn.model = _learn.model.cpu()
    except Exception:
        pass


learn = None
probs = None

_reusable_test_dl = None
_reusable_tfms_sig = None

_pil_cache = {}
_cached_items = _PILCachedItems(test_fnames, cache=_pil_cache)

if len(existing_models) > 0:
    for mp in existing_models:
        _learn = load_learner(fname=mp, cpu=False, pickle_module=dill)

        _sig = _tfms_signature(_learn)
        if _reusable_test_dl is None or _reusable_tfms_sig != _sig:
            _reusable_test_dl = _build_fast_test_dl(_learn, _cached_items)
            _reusable_tfms_sig = _sig

        with _learn.no_bar(), _learn.no_logging():
            probs_temp = _predict_probs_tta_fast_cached(
                _learn, _reusable_test_dl, tfms_sig=_reusable_tfms_sig, n_tta=4
            )

        if probs is None:
            probs = probs_temp
            learn = _learn  # keep first learner for vocab
        else:
            probs += probs_temp
            _free_learner(_learn)
            del _learn, probs_temp

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
else:
    train_df = pd.read_csv(TRAIN_CSV)

    chains = train_df["chain"].astype(str).to_numpy()
    images = train_df["image"].astype(str).to_numpy()
    filepaths = TRAIN_IMG_DIR.as_posix() + "/" + chains + "/" + images
    train_df = train_df.assign(filepath=filepaths)

    from os.path import exists as _os_exists

    exists_mask = np.fromiter(
        (_os_exists(fp) for fp in filepaths), dtype=bool, count=len(filepaths)
    )
    train_df = train_df.loc[exists_mask].reset_index(drop=True)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader("filepath"),
        get_y=ColReader("hotel_id"),
        splitter=RandomSplitter(valid_pct=0.1, seed=42),
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
    )
    dls = dblock.dataloaders(train_df, bs=64, num_workers=2)
    learn = vision_learner(dls, resnet34, metrics=accuracy)

    with learn.no_bar(), learn.no_logging():
        learn.fine_tune(1)

    test_dl = _build_fast_test_dl(learn, _cached_items)
    sig = _tfms_signature(learn)
    with learn.no_bar(), learn.no_logging():
        probs = _predict_probs_tta_fast_cached(learn, test_dl, tfms_sig=sig, n_tta=4)

probs.shape



## === cell 6
preds_idx = probs.topk(5, dim=1)[1].detach().cpu().numpy()

vocab = learn.dls.vocab
vocab_arr = np.asarray(vocab, dtype=object).astype(str, copy=False)
preds = [" ".join(vocab_arr[row]) for row in preds_idx]

preds[:3], len(preds)



## === cell 7
out = submission.copy()
out["hotel_id"] = preds

out = out[["image", "hotel_id"]]
out.to_csv("submission.csv", index=False)

out.head()
