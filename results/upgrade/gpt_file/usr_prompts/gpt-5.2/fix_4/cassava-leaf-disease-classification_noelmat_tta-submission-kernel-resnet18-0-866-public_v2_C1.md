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
import numpy as np



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


def stratified_splitter_from_df(df, valid_pct=0.2, seed=42, label_col="label"):
    rng = np.random.RandomState(seed)
    train_idxs, valid_idxs = [], []
    y = df[label_col].values
    for c in np.unique(y):
        idxs = np.where(y == c)[0]
        rng.shuffle(idxs)
        n_valid = max(1, int(round(len(idxs) * valid_pct)))
        valid_idxs.extend(idxs[:n_valid].tolist())
        train_idxs.extend(idxs[n_valid:].tolist())
    train_idxs = L(train_idxs)
    valid_idxs = L(valid_idxs)
    return train_idxs, valid_idxs


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

    splitter = stratified_splitter_from_df(
        train_df, valid_pct=0.2, seed=seed, label_col="label"
    )

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
/tmp/ipykernel_55/1899418621.py in <cell line: 0>()
      1 set_seed(42, reproducible=True)
      2 
----> 3 dls = get_data(labels, train_df, bs=128, presize=384, seed=42)
      4 

/tmp/ipykernel_55/368050489.py in get_data(labels, train_df, bs, presize, resize, seed)
     55         item_tfms=[Resize(presize)],
     56         batch_tfms=[*comp, Normalize.from_stats(*imagenet_stats)],
---> 57     ).dataloaders(path / "train_images", bs=bs, num_workers=8)
     58 
     59 

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    155         **kwargs
    156     ) -> DataLoaders:
--> 157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in datasets(self, source, verbose)
    145         self.source = source                     ; pv(f"Collecting items from {source}", verbose)
    146         items = (self.get_items or noop)(source) ; pv(f"Found {len(items)} items", verbose)
--> 147         splits = (self.splitter or RandomSplitter())(items)
    148         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
    149         return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)

TypeError: 'tuple' object is not callable

## === cell 4
learn = cnn_learner(dls, models.resnet18, metrics=[accuracy], pretrained=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2690965047.py in <cell line: 0>()
----> 1 learn = cnn_learner(dls, models.resnet18, metrics=[accuracy], pretrained=True)
      2 

NameError: name 'dls' is not defined

## === cell 5
sample_sub = pd.read_csv(path / "sample_submission.csv")
test_files = [path / "test_images" / fn for fn in sample_sub["image_id"].tolist()]

model_candidates = [
    Path("/kaggle/input/cassava-starter/models"),
    Path("/kaggle/input/casava-starter/models"),  # keep original typo path as fallback
]

loaded = False
for mdir in model_candidates:
    try:
        if mdir.exists():
            pths = list(mdir.glob("*.pth"))
            if len(pths) > 0:
                src = pths[0]
                learn.path.mkdir(parents=True, exist_ok=True)
                (learn.path / learn.model_dir).mkdir(parents=True, exist_ok=True)
                dst = learn.path / learn.model_dir / src.name
                if not dst.exists():
                    import shutil

                    shutil.copyfile(src, dst)
                learn.load(dst.stem)
                loaded = True
                break
    except Exception:
        pass

if not loaded:
    set_seed(42, reproducible=True)
    learn.fit_one_cycle(1, 3e-3)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2485307917.py in <cell line: 0>()
     33 if not loaded:
     34     set_seed(42, reproducible=True)
---> 35     learn.fit_one_cycle(1, 3e-3)
     36 

NameError: name 'learn' is not defined

## === cell 6
test_dl = dls.test_dl(test_files)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3643398007.py in <cell line: 0>()
----> 1 test_dl = dls.test_dl(test_files)
      2 

NameError: name 'dls' is not defined

## === cell 7
preds = learn.tta(dl=test_dl)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/430561149.py in <cell line: 0>()
----> 1 preds = learn.tta(dl=test_dl)
      2 

NameError: name 'learn' is not defined

## === cell 8
pred_labels = torch.argmax(preds[0], dim=1).cpu().numpy().astype(int)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3539167996.py in <cell line: 0>()
----> 1 pred_labels = torch.argmax(preds[0], dim=1).cpu().numpy().astype(int)
      2 

NameError: name 'preds' is not defined

## === cell 9
sample_sub["label"] = pred_labels.astype(int)

if sample_sub["label"].isna().any():
    sample_sub["label"] = sample_sub["label"].fillna(0).astype(int)

sample_sub.to_csv("submission.csv", index=False)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/220802672.py in <cell line: 0>()
      1 # Build submission aligned to sample_submission.csv order (already matched via test_files construction).
----> 2 sample_sub["label"] = pred_labels.astype(int)
      3 
      4 # Safety checks to always write a valid submission
      5 if sample_sub["label"].isna().any():

NameError: name 'pred_labels' is not defined

## === cell 10
sample_sub
