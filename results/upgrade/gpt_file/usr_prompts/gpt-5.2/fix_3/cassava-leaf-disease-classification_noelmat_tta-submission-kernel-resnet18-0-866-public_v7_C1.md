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

0.8878815352070112

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.62706) has done: 'The run is failing because it tries to load a pretrained `.pth` file from a Kaggle dataset (`/kaggle/input/casava-starter/...`) that is not available, which leaves `predictions` empty and causes downstream tensor/variable errors. I keep the same fastai DataBlock/cnn_learner/ResNet50 + TTA prediction logic, but add a safe fallback: if the external weights aren’t present, train the learner briefly on the provided `train_images` and then run TTA on the test set. I also make the submission creation robust by converting filenames to a list (not an exhausted `map`) and ensuring lengths align, so `submission.csv` is always written with the correct columns. These changes are necessary to produce a valid submission end-to-end and should yield a non-trivial accuracy score versus a broken/no-submission pipeline.'

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import pandas as pd, numpy as np, json
from functools import partial



## === cell 1
path = Path("/kaggle/input/cassava-leaf-disease-classification/")
path.ls()



## === cell 2
train_df = pd.read_csv(path / "train.csv")
with open(path / "label_num_to_disease_map.json") as f:
    label_dict = json.load(f)
label_dict = {int(k): v for k, v in label_dict.items()}

df = train_df.set_index("image_id")
labels = df.to_dict()["label"]


def get_label(labels, x):
    x = Path(x)
    return labels[x.name]


def get_data(labels, bs=64, presize=500, resize=384, valid_pct=0.2, seed=42):
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

    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda p: get_image_files(p),
        get_y=partial(get_label, labels),
        splitter=GrandparentSplitter(valid_name="..") if False else None,
        item_tfms=[Resize(presize)],
        batch_tfms=[*comp, Normalize.from_stats(*imagenet_stats)],
    ).dataloaders(
        path / "train_images",
        bs=bs,
        num_workers=8,
        splitter=ColSplitter(col="is_valid"),
    )




## === cell 3
set_seed(42, reproducible=True)

train_df2 = train_df.copy()
splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx = np.arange(len(train_df2))
y = train_df2["label"].values
train_idx, valid_idx = next(splits.split(idx, y))
is_valid = np.zeros(len(train_df2), dtype=bool)
is_valid[valid_idx] = True
train_df2["is_valid"] = is_valid

labels2 = train_df2.set_index("image_id")["label"].to_dict()


def get_data_from_df(train_df, labels, bs=64, presize=384, resize=384):
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

    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader("image_id", pref=str(path / "train_images") + "/"),
        get_y=ColReader("label"),
        splitter=ColSplitter(col="is_valid"),
        item_tfms=[Resize(presize)],
        batch_tfms=[*comp, Normalize.from_stats(*imagenet_stats)],
    ).dataloaders(train_df, bs=bs, num_workers=8)


dls = get_data_from_df(train_df2, labels2, bs=128, presize=384, resize=384)

learn = cnn_learner(dls, models.resnet50, metrics=[accuracy], pretrained=True)

test_files = get_image_files(path / "test_images")
predictions = []

weights_base = Path("/kaggle/input/casava-starter/models")
for fold in range(1):
    wfile = weights_base / f"resnet50-full-fold_{fold}.pth"
    if wfile.exists():
        state = torch.load(wfile, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            learn.model.load_state_dict(state["model"])
        elif isinstance(state, dict):
            learn.model.load_state_dict(state)
        else:
            learn.model.load_state_dict(state)
    else:
        learn.fine_tune(4, base_lr=3e-3)

    test_dl = dls.test_dl(test_files)
    preds, _ = learn.tta(dl=test_dl, n=6)
    predictions.append(preds)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/126677336.py in <cell line: 0>()
      4 train_df2 = train_df.copy()
      5 # Stratified split
----> 6 splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
      7 idx = np.arange(len(train_df2))
      8 y = train_df2["label"].values

NameError: name 'StratifiedShuffleSplit' is not defined

## === cell 4
if len(predictions) == 0:
    raise RuntimeError(
        "No predictions were generated; check model loading/training step."
    )

mean_preds = torch.mean(torch.stack(predictions), dim=0)
preds = torch.argmax(mean_preds, dim=1)

test_fn = [p.name for p in test_files]

assert len(test_fn) == len(
    preds
), f"Mismatch: {len(test_fn)} filenames vs {len(preds)} predictions"



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/40060940.py in <cell line: 0>()
----> 1 if len(predictions) == 0:
      2     raise RuntimeError(
      3         "No predictions were generated; check model loading/training step."
      4     )
      5 

NameError: name 'predictions' is not defined

## === cell 5
submission = pd.DataFrame(
    {"image_id": test_fn, "label": preds.cpu().numpy().astype(int)}
)
submission.to_csv("submission.csv", index=False)
submission.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3571970132.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"image_id": test_fn, "label": preds.cpu().numpy().astype(int)}
      3 )
      4 submission.to_csv("submission.csv", index=False)
      5 submission.head()

NameError: name 'test_fn' is not defined

## === cell 6
submission

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
