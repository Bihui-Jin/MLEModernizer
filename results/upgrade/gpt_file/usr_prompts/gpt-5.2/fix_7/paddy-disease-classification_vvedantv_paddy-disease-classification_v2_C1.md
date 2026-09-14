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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.8894009216589862

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path

if False:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))

comp = "paddy-disease-classification"
base = Path("/kaggle/input")

for p in (
    base / comp,
    base / comp / comp,
    base / "paddy-disease-classification",
    base / "paddy-disease-classification" / "paddy-disease-classification",
):
    if p.exists():
        path = p
        break
else:
    raise AssertionError(f"Dataset path not found. Tried under: {base}")

path




## === cell 1
from fastai.vision.all import *

set_seed(42, reproducible=True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    torch.backends.cudnn.benchmark = False

if False:
    path.ls()




## === cell 2
trn_path = path / "train_images"

files = get_image_files(trn_path)
len(files), files[0]




## === cell 3
if False:
    img = PILImage.create(files[0])
    print(img.size)
    img.to_thumb(128)




## === cell 4
if False:
    from fastcore.parallel import *

    def f(o):
        return PILImage.create(o).size

    sizes = parallel(f, files, n_workers=min(8, os.cpu_count() or 1))
    pd.Series(sizes).value_counts()




## === cell 5
n_cpu = os.cpu_count() or 1

item_tfms = Resize(128, method="squish")
batch_tfms = aug_transforms(size=128, min_scale=0.75)

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

worker_cnt = min(
    4, n_cpu
)  # conservative: reduces multiprocessing overhead while keeping parallel decode
dls = dblock.dataloaders(
    trn_path,
    bs=64,
    num_workers=worker_cnt,
    prefetch_factor=4 if worker_cnt > 0 else None,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(worker_cnt > 0),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
).cache()

if False:
    dls.show_batch(max_n=6)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1863058404.py in <cell line: 0>()
     27     persistent_workers=(worker_cnt > 0),
     28     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
---> 29 ).cache()
     30 
     31 if False:

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __getattr__(self, k)
    455         return res if is_indexer(it) else list(zip(*res))
    456 
--> 457     def __getattr__(self,k): return gather_attrs(self, k, 'tls')
    458     def __dir__(self): return super().__dir__() + gather_attr_names(self, 'tls')
    459     def __len__(self): return len(self.tls[0])

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in gather_attrs(o, k, nm)
    211     att = getattr(o,nm)
    212     res = [t for t in att.attrgot(k) if t is not None]
--> 213     if not res: raise AttributeError(k)
    214     return res[0] if len(res)==1 else L(res)
    215 

AttributeError: cache

## === cell 6
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")

if torch.cuda.is_available():
    learn = learn.to_fp16()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/762767824.py in <cell line: 0>()
----> 1 learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")
      2 
      3 # Speed/correctness: only use fp16 on GPU; on CPU it can be slower/unsupported and risks slowdown/timeouts.
      4 if torch.cuda.is_available():
      5     learn = learn.to_fp16()

NameError: name 'dls' is not defined

## === cell 7
if False:
    learn.lr_find(suggest_funcs=(valley, slide))




## === cell 8
learn.fine_tune(3, 0.01)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/371260437.py in <cell line: 0>()
----> 1 learn.fine_tune(3, 0.01)
      2 
      3 

NameError: name 'learn' is not defined

## === cell 9
ss = pd.read_csv(path / "sample_submission.csv")
ss.head(), ss.shape




## === cell 10
tst_files = get_image_files(path / "test_images")
tst_files = sorted(tst_files, key=lambda p: p.name)

tst_dl = dls.test_dl(
    tst_files,
    bs=128,
    num_workers=worker_cnt,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(worker_cnt > 0),
)

len(tst_files), tst_files[0].name, tst_files[-1].name




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2453276479.py in <cell line: 0>()
      3 tst_files = sorted(tst_files, key=lambda p: p.name)
      4 
----> 5 tst_dl = dls.test_dl(
      6     tst_files,
      7     bs=128,

NameError: name 'dls' is not defined

## === cell 11
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs[:10], probs.shape




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/925903878.py in <cell line: 0>()
----> 1 probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
      2 idxs[:10], probs.shape
      3 
      4 

NameError: name 'learn' is not defined

## === cell 12
dls.vocab




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1985098568.py in <cell line: 0>()
----> 1 dls.vocab
      2 
      3 

NameError: name 'dls' is not defined

## === cell 13
vocab = np.array(dls.vocab, dtype=object)
preds_labels = vocab[idxs.cpu().numpy()].tolist()
len(preds_labels), preds_labels[:10]




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1704777547.py in <cell line: 0>()
----> 1 vocab = np.array(dls.vocab, dtype=object)
      2 preds_labels = vocab[idxs.cpu().numpy()].tolist()
      3 len(preds_labels), preds_labels[:10]
      4 
      5 

NameError: name 'dls' is not defined

## === cell 14
pred_df = pd.DataFrame({"image_id": [p.name for p in tst_files], "label": preds_labels})

sub = ss.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")

missing = sub["label"].isna().sum()
assert (
    missing == 0
), f"Missing predictions for {missing} rows; check test file discovery / merge."

sub.to_csv("submission.csv", index=False)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/746167015.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame({"image_id": [p.name for p in tst_files], "label": preds_labels})
      2 
      3 sub = ss.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")
      4 
      5 missing = sub["label"].isna().sum()

NameError: name 'preds_labels' is not defined
