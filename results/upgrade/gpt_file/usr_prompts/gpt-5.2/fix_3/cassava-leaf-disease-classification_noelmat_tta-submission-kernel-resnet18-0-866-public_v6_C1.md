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

0.8703535811423391

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.7003) has done: 'The run fails because it tries to load pretrained fold checkpoints from a non-existent Kaggle dataset path, leaving `predictions` empty and breaking stacking/submission creation. I keep the same fastai DataBlock + ResNet18 + TTA inference logic, but add a minimal fallback: if the external fold weights aren’t available, train the learner on the provided `train_images` for a small number of epochs and then run TTA once to generate test predictions. I also fix missing imports (`pandas`, `json`) and ensure `image_id` alignment and a properly formatted `submission.csv` is always written. These changes are strictly to make the notebook run end-to-end and produce a valid CSV, with the smallest training needed to yield a meaningful score.'

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import pandas as pd
import json



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


def get_data(labels, train_df, bs=64, presize=500, resize=384, valid_pct=0.2, seed=42):
    splitter = ColSplitter(col="is_valid")

    df2 = train_df.copy()
    df2["is_valid"] = False
    rng = np.random.RandomState(seed)
    for lbl, g in df2.groupby("label"):
        idx = g.index.to_numpy()
        n_valid = max(1, int(round(valid_pct * len(idx))))
        valid_idx = rng.choice(idx, size=n_valid, replace=False)
        df2.loc[valid_idx, "is_valid"] = True

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
        get_items=lambda p: df2["image_id"].tolist(),
        get_x=lambda fn: path / "train_images" / fn,
        get_y=partial(get_label, labels),
        splitter=splitter,
        item_tfms=[Resize(presize)],
        batch_tfms=[*comp, Normalize.from_stats(*imagenet_stats)],
    ).dataloaders(path, bs=bs, num_workers=8)




## === cell 3
set_seed(42, reproducible=True)

dls = get_data(
    labels, train_df, bs=128, presize=384, resize=384, valid_pct=0.2, seed=42
)

learn = cnn_learner(dls, models.resnet18, metrics=[accuracy], pretrained=True)

sample_sub = pd.read_csv(path / "sample_submission.csv")
test_files = [path / "test_images" / fn for fn in sample_sub["image_id"].tolist()]
test_dl = dls.test_dl(test_files)

predictions = []

ext_model_dir = Path("/kaggle/input/casava-starter/models")
has_external = ext_model_dir.exists()

if has_external:
    for fold in range(5):
        model_path = ext_model_dir / f"resnet18-full-fold_{fold}.pth"
        if model_path.exists():
            learn.load(str(model_path.with_suffix("")))
            preds, _ = learn.tta(dl=test_dl, n=6)
            predictions.append(preds)

if len(predictions) == 0:
    learn.fit_one_cycle(5, 1e-3)
    preds, _ = learn.tta(dl=test_dl, n=6)
    predictions = [preds]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/867500955.py in <cell line: 0>()
      1 set_seed(42, reproducible=True)
      2 
----> 3 dls = get_data(
      4     labels, train_df, bs=128, presize=384, resize=384, valid_pct=0.2, seed=42
      5 )

/tmp/ipykernel_55/2743096869.py in get_data(labels, train_df, bs, presize, resize, valid_pct, seed)
     49         item_tfms=[Resize(presize)],
     50         batch_tfms=[*comp, Normalize.from_stats(*imagenet_stats)],
---> 51     ).dataloaders(path, bs=bs, num_workers=8)
     52 
     53 

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

/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py in _inner(o)
    163     "Split `items` (supposed to be a dataframe) by value in `col`"
    164     def _inner(o):
--> 165         assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
    166         c = o.iloc[:,col] if isinstance(col, int) else o[col]
    167         if on is None:      valid_idx = c.values.astype('bool')

AssertionError: ColSplitter only works when your items are a pandas DataFrame

## === cell 4
preds = torch.argmax(torch.mean(torch.stack(predictions), dim=0), dim=1).cpu().numpy()
test_fn = [p.name for p in test_files]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2111094861.py in <cell line: 0>()
----> 1 preds = torch.argmax(torch.mean(torch.stack(predictions), dim=0), dim=1).cpu().numpy()
      2 test_fn = [p.name for p in test_files]
      3 

NameError: name 'predictions' is not defined

## === cell 5
submission = pd.DataFrame({"image_id": test_fn, "label": preds})

submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")

submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == len(test_files)
assert list(submission.columns) == ["image_id", "label"]
submission.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1228816947.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_fn, "label": preds})
      2 
      3 # Enforce exact sample submission order (defensive alignment)
      4 submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
      5 

NameError: name 'test_fn' is not defined

## === cell 6
submission

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
