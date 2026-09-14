# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

_TEST_FILES = get_image_files(path / "test_images")
test_files = _TEST_FILES

test_dl_kwargs = dict(
    bs=128,
    num_workers=dls.num_workers,
    pin_memory=True,
    shuffle=False,
)
try:
    test_dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
except Exception:
    pass

test_dl = dls.test_dl(test_files, **test_dl_kwargs)

predictions = []

ext_ckpt_dir = Path("/kaggle/input/resnet50/models")
loaded_any = False

learn.model.eval()
device = learn.dls.device


def _predict_tta_folds_avg_online(learn, dl, ckpt_paths, n_tta=6, n_classes=5):
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

    n_folds_loaded = 0
    with torch.inference_mode():
        for ckpt in ckpt_paths:
            if not ckpt.exists():
                continue
            learn.load(ckpt.with_suffix(""))  # fastai expects path without .pth
            mdl.eval()
            n_folds_loaded += 1

            for _ in range(n_tta):
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

    sum_preds.div_(n_tta * n_folds_loaded)
    return sum_preds.detach().cpu()


_TTA_N = 6
ckpt_paths = [ext_ckpt_dir / f"resnet50-full-fold_{fold}.pth" for fold in range(3)]
avg_preds = _predict_tta_folds_avg_online(
    learn, test_dl, ckpt_paths, n_tta=_TTA_N, n_classes=5
)

if avg_preds is None:
    loaded_any = False
    learn.fine_tune(5, base_lr=3e-3)
    avg_preds = _predict_tta_folds_avg_online(
        learn, test_dl, [Path("dummy_nonexistent.pth")], n_tta=0, n_classes=5
    )

    def _predict_tta_single(learn, dl, n_tta=6, n_classes=5):
        mdl = learn.model
        dev = learn.dls.device
        use_cuda = torch.cuda.is_available() and (
            str(dev).startswith("cuda") or getattr(dev, "type", "") == "cuda"
        )
        if use_cuda:
            mdl = mdl.to(memory_format=torch.channels_last)

        n_items = len(dl.dataset)
        sum_preds_dev = torch.zeros(
            (n_items, n_classes), dtype=torch.float32, device=dev
        )
        mdl_fwd = mdl.__call__
        with torch.inference_mode():
            for _ in range(n_tta):
                start = 0
                for b in dl:
                    xb = b[0] if isinstance(b, (tuple, list)) else b
                    bs = xb.shape[0]
                    xb = xb.to(dev, non_blocking=True)
                    if use_cuda:
                        xb = xb.contiguous(memory_format=torch.channels_last)
                    out = mdl_fwd(xb).to(dtype=torch.float32)
                    sum_preds_dev[start : start + bs].add_(out)
                    start += bs
        sum_preds_dev.div_(n_tta)
        return sum_preds_dev.detach().cpu()

    avg_preds = _predict_tta_single(learn, test_dl, n_tta=_TTA_N, n_classes=5)
else:
    loaded_any = True



## === cell 4
pred_labels = avg_preds.argmax(dim=1).cpu().numpy()

sample_sub = pd.read_csv(path / "sample_submission.csv")

test_fn = [p.name for p in test_files]
pred_map = dict(zip(test_fn, pred_labels.astype(int)))
ordered_labels = sample_sub["image_id"].map(pred_map).astype(int).values



## === cell 5
submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": ordered_labels}
)
submission = submission[["image_id", "label"]]
assert len(submission) == len(sample_sub)
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 6
submission
