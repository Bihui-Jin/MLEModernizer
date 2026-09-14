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

0.8644605621033545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.64537) has done: 'Your code doesn’t yield a Kaggle score because it’s likely failing before writing a valid `submission.csv`: `pd`/`json` aren’t imported, the model path probably doesn’t exist (`casava-starter` typo), and the submission currently writes tensors/generators instead of plain Python lists aligned to `sample_submission.csv`. I make minimal fixes: add the missing imports, robustly load the model if the file exists (otherwise fall back to training briefly so a valid submission is always produced), and ensure predictions are converted to `int` and ordered exactly like `sample_submission.csv`. These changes keep your core fastai DataBlock/cnn_learner/tta approach intact while ensuring an end-to-end run and a valid submission file. This should move the score toward your target by (a) actually producing a submission and (b) avoiding misalignment/format issues that can tank accuracy.'

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import pandas as pd
import json
from pathlib import Path



## === cell 1
path = Path("/kaggle/input/cassava-leaf-disease-classification/")
path.ls()



## === cell 2
train_df = pd.read_csv(path / "train.csv")
train_df.head()

with open(path / "label_num_to_disease_map.json") as f:
    label_dict = json.load(f)
label_dict = {int(k): v for k, v in label_dict.items()}

df = train_df.set_index("image_id")
labels = df.to_dict()["label"]


def get_label(labels, x):
    x = Path(x)
    return labels[x.name]


def get_data(labels, train_df, bs=64, presize=500, resize=384, seed=42):
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

    splitter = RandomSplitter(seed=seed, stratify=train_df["label"].values)

    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda p: get_image_files(p),
        get_y=partial(get_label, labels),
        splitter=splitter,
        item_tfms=[Resize(presize)],
        batch_tfms=[*comp, Normalize.from_stats(*imagenet_stats)],
    ).dataloaders(path / "train_images", bs=bs, num_workers=8)




## === cell 3
set_seed(42, reproducible=True)

dls = get_data(labels, train_df, bs=128, presize=384, seed=42)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1093945567.py in <cell line: 0>()
      2 set_seed(42, reproducible=True)
      3 
----> 4 dls = get_data(labels, train_df, bs=128, presize=384, seed=42)
      5 

/tmp/ipykernel_55/811220554.py in get_data(labels, train_df, bs, presize, resize, seed)
     29     # Change (score-relevant, minimal): use a stratified, reproducible split to stabilize training signal
     30     # and usually improve accuracy vs a purely random split on imbalanced labels.
---> 31     splitter = RandomSplitter(seed=seed, stratify=train_df["label"].values)
     32 
     33     return DataBlock(

TypeError: RandomSplitter() got an unexpected keyword argument 'stratify'

## === cell 4
learn = cnn_learner(dls, models.resnet18, metrics=[accuracy], pretrained=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3887705118.py in <cell line: 0>()
      1 # Change (score-relevant): enable pretrained weights; training from scratch typically underfits here.
----> 2 learn = cnn_learner(dls, models.resnet18, metrics=[accuracy], pretrained=True)
      3 

NameError: name 'dls' is not defined

## === cell 5
test_files = get_image_files(path / "test_images")



## === cell 6
model_candidates = [
    Path("/kaggle/input/cassava-starter/models/resnet18-full"),
    Path(
        "/kaggle/input/casava-starter/models/resnet18-full"
    ),  # keep original typo path as fallback
]
loaded = False
for mpath in model_candidates:
    try:
        if mpath.exists():
            learn.load(mpath)
            loaded = True
            break
    except Exception:
        pass

if not loaded:
    set_seed(42, reproducible=True)
    learn.fit_one_cycle(1, 3e-3)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4196065324.py in <cell line: 0>()
     18 if not loaded:
     19     set_seed(42, reproducible=True)
---> 20     learn.fit_one_cycle(1, 3e-3)
     21 

NameError: name 'learn' is not defined

## === cell 7
test_dl = dls.test_dl(test_files)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3643398007.py in <cell line: 0>()
----> 1 test_dl = dls.test_dl(test_files)
      2 

NameError: name 'dls' is not defined

## === cell 8
preds = learn.tta(dl=test_dl)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/430561149.py in <cell line: 0>()
----> 1 preds = learn.tta(dl=test_dl)
      2 

NameError: name 'learn' is not defined

## === cell 9
pred_labels = torch.argmax(preds[0], dim=1).cpu().numpy().astype(int)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3539167996.py in <cell line: 0>()
----> 1 pred_labels = torch.argmax(preds[0], dim=1).cpu().numpy().astype(int)
      2 

NameError: name 'preds' is not defined

## === cell 10
sample_sub = pd.read_csv(path / "sample_submission.csv")
pred_map = {f.name: int(p) for f, p in zip(test_files, pred_labels)}
sample_sub["label"] = sample_sub["image_id"].map(pred_map).astype(int)

if sample_sub["label"].isna().any():
    sample_sub["label"] = sample_sub["label"].fillna(0).astype(int)

sample_sub.to_csv("submission.csv", index=False)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/248122645.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(path / "sample_submission.csv")
----> 2 pred_map = {f.name: int(p) for f, p in zip(test_files, pred_labels)}
      3 sample_sub["label"] = sample_sub["image_id"].map(pred_map).astype(int)
      4 
      5 # Safety: if any images are missing (shouldn't happen), fill with a valid class to ensure a valid submission.

NameError: name 'pred_labels' is not defined

## === cell 11
sample_sub
