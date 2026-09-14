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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8847083711090964

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.63042) has done: 'The crash is caused by trying to load pretrained fold weights from `/kaggle/input/resnet50/...` which don’t exist in this environment; that prevents any predictions from being generated and causes downstream tensor/variable errors. I keep the same fastai `resnet50` inference core, but add a safe fallback: if fold checkpoints are missing, train the same model briefly on the provided `train.csv` so we can still produce a valid submission. I also fix missing imports (`pandas`, `json`) and ensure predictions are generated deterministically and the submission is written with the exact required columns and `.csv` suffix. These changes are minimal and focused on making the pipeline run end-to-end and yield a reasonable accuracy-based submission.'

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import pandas as pd
import json
from functools import partial
import os



## === cell 1
path = Path("/kaggle/input/cassava-leaf-disease-classification/")

_ = path  # keep variable alive; no side effects



## === cell 2
train_df = pd.read_csv(path / "train.csv")
with open(path / "label_num_to_disease_map.json") as f:
    label_dict = json.load(f)
label_dict = {int(k): v for k, v in label_dict.items()}

df = train_df.set_index("image_id")
labels = df["label"].to_dict()


def get_label(labels, x):
    return labels[x.name]


def _build_is_valid_map(_train_df, seed=42, frac=0.2):
    _df = _train_df.copy()
    set_seed(seed, reproducible=True)
    valid_df = _df.groupby("label", group_keys=False).sample(
        frac=frac, random_state=seed
    )
    valid_set = set(valid_df["image_id"].values.tolist())
    return {img_id: (img_id in valid_set) for img_id in _df["image_id"].values.tolist()}


_IS_VALID_MAP = _build_is_valid_map(train_df, seed=42, frac=0.2)

_TRAIN_FILES = None


def get_data(labels, bs=64, presize=500, resize=384):
    tfms = [
        Rotate(90),
        Warp(magnitude=0.4, p=1.0),
        Zoom(min_zoom=0.9, max_zoom=1.3),
        Brightness(max_lighting=0.5),
        Flip(),
        Contrast(),
        Resize(resize),
    ]
    comp = setup_aug_tfms(tfms)

    is_valid_map = _IS_VALID_MAP

    def _get_items(p):
        global _TRAIN_FILES
        if _TRAIN_FILES is None:
            _TRAIN_FILES = get_image_files(p)
        return _TRAIN_FILES

    def _splitter(items):
        mget = is_valid_map.get
        train, valid = [], []
        for i, o in enumerate(items):
            if mget(o.name, False):
                valid.append(i)
            else:
                train.append(i)
        return train, valid

    ncpu = os.cpu_count() or 2
    nw = min(8, max(2, ncpu // 2))

    dl_kwargs = dict(num_workers=nw, pin_memory=True)
    try:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    except Exception:
        pass

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=_get_items,
        get_y=partial(get_label, labels),
        splitter=_splitter,
        item_tfms=[Resize(presize)],
        batch_tfms=[*comp, Normalize.from_stats(*imagenet_stats)],
    ).dataloaders(path / "train_images", bs=bs, **dl_kwargs)

    return dls




## === cell 3
set_seed(42, reproducible=True)

torch.backends.cudnn.benchmark = True

dls = get_data(labels, bs=128, presize=384, resize=384)
learn = cnn_learner(dls, models.resnet50, metrics=[accuracy], pretrained=True)

sample_sub = pd.read_csv(path / "sample_submission.csv")
sub_ids = sample_sub["image_id"].tolist()

test_files = [(path / "test_images" / fn) for fn in sub_ids]

missing = [p.name for p in test_files if not p.exists()]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images (first 5): {missing[:5]}"
    )

ncpu = os.cpu_count() or 2
nw = min(8, max(2, ncpu // 2))
test_dl_kwargs = dict(
    bs=128,
    num_workers=nw,
    pin_memory=True,
    shuffle=False,
)
try:
    test_dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
except Exception:
    pass

test_block = DataBlock(
    blocks=(ImageBlock,),
    get_items=lambda p: test_files,
    item_tfms=[Resize(384)],
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)
test_dls = test_block.dataloaders(path / "test_images", **test_dl_kwargs)
test_dl = test_dls.train  # single dataloader is fine; no labels/split needed

ext_ckpt_dir = Path("/kaggle/input/resnet50/models")

learn.model.eval()
device = learn.dls.device


def _predict_folds_avg_singlepass(learn, dl, ckpt_paths, n_classes=5):
    mdl = learn.model
    dev = learn.dls.device

    use_cuda = torch.cuda.is_available() and (
        str(dev).startswith("cuda") or getattr(dev, "type", "") == "cuda"
    )
    if use_cuda:
        mdl = mdl.to(memory_format=torch.channels_last)

    n_items = len(dl.dataset)
    sum_preds = torch.zeros((n_items, n_classes), dtype=torch.float32, device=dev)

    mdl_fwd = mdl.__call__
    to_dev = dev
    float32 = torch.float32

    existing = [p for p in ckpt_paths if p.exists()]
    n_folds_loaded = 0

    with torch.inference_mode():
        for ckpt in existing:
            learn.load(ckpt.with_suffix(""))  # fastai expects path without .pth
            mdl.eval()
            n_folds_loaded += 1

            start = 0
            for b in dl:
                xb = b[0] if isinstance(b, (tuple, list)) else b
                bs = xb.shape[0]

                xb = xb.to(to_dev, non_blocking=True)
                if use_cuda:
                    xb = xb.contiguous(memory_format=torch.channels_last)

                out = mdl_fwd(xb).to(dtype=float32)
                sum_preds[start : start + bs].add_(out)
                start += bs

    if n_folds_loaded == 0:
        return None

    sum_preds.div_(n_folds_loaded)
    return sum_preds.detach().cpu()


def _predict_single(learn, dl, n_classes=5):
    mdl = learn.model
    dev = learn.dls.device
    use_cuda = torch.cuda.is_available() and (
        str(dev).startswith("cuda") or getattr(dev, "type", "") == "cuda"
    )
    if use_cuda:
        mdl = mdl.to(memory_format=torch.channels_last)

    n_items = len(dl.dataset)
    preds_dev = torch.zeros((n_items, n_classes), dtype=torch.float32, device=dev)
    mdl_fwd = mdl.__call__

    with torch.inference_mode():
        start = 0
        for b in dl:
            xb = b[0] if isinstance(b, (tuple, list)) else b
            bs = xb.shape[0]
            xb = xb.to(dev, non_blocking=True)
            if use_cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)
            out = mdl_fwd(xb).to(dtype=torch.float32)
            preds_dev[start : start + bs].copy_(out)
            start += bs

    return preds_dev.detach().cpu()


_TTA_N = 6  # kept for compatibility with original variables; not used

ckpt_paths = [ext_ckpt_dir / f"resnet50-full-fold_{fold}.pth" for fold in range(3)]
avg_preds = _predict_folds_avg_singlepass(learn, test_dl, ckpt_paths, n_classes=5)

if avg_preds is None:
    learn.fine_tune(5, base_lr=3e-3)
    avg_preds = _predict_single(learn, test_dl, n_classes=5)

if avg_preds is None or len(avg_preds) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length mismatch: preds={None if avg_preds is None else len(avg_preds)} vs submission={len(sample_sub)}"
    )



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2845640084.py in <cell line: 0>()
    133 # Sanity check: prediction count must match submission rows
    134 if avg_preds is None or len(avg_preds) != len(sample_sub):
--> 135     raise RuntimeError(
    136         f"Prediction length mismatch: preds={None if avg_preds is None else len(avg_preds)} vs submission={len(sample_sub)}"
    137     )

RuntimeError: Prediction length mismatch: preds=2141 vs submission=2676

## === cell 4
pred_labels = avg_preds.argmax(dim=1).cpu().numpy().astype(int)

ordered_labels = pred_labels
assert len(ordered_labels) == len(sample_sub)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/659230030.py in <cell line: 0>()
      3 # Since test_files are already in sample_submission order, no mapping is needed.
      4 ordered_labels = pred_labels
----> 5 assert len(ordered_labels) == len(sample_sub)
      6 

AssertionError: 

## === cell 5
submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": ordered_labels}
)
submission = submission[["image_id", "label"]]
assert len(submission) == len(sample_sub)
submission.to_csv("submission.csv", index=False)

submission.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/431708559.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(
      2     {"image_id": sample_sub["image_id"].values, "label": ordered_labels}
      3 )
      4 submission = submission[["image_id", "label"]]
      5 assert len(submission) == len(sample_sub)

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in __init__(self, data, index, columns, dtype, copy)
    590 def __init__(self:pd.DataFrame, data=None, index=None, columns=None, dtype=None, copy=None):
    591     if data is not None and isinstance(data, Tensor): data = to_np(data)
--> 592     self._old_init(data, index=index, columns=columns, dtype=dtype, copy=copy)
    593 
    594 # %% ../nbs/00_torch_core.ipynb 153

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
