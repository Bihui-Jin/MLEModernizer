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

3.12

# 3. Installed packages



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

0.88479

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import pandas as pd

FASTAI_AVAILABLE = True
try:
    from fastai.vision.all import *
    import timm  # noqa: F401
except Exception as e:
    FASTAI_AVAILABLE = False
    FASTAI_IMPORT_ERROR = repr(e)

if FASTAI_AVAILABLE:
    set_seed(42, reproducible=True)
    import torch

    _cpu = os.cpu_count() or 2
    torch.set_num_threads(min(8, _cpu))
    torch.set_num_interop_threads(1)

    torch.backends.cudnn.benchmark = True

comp = "paddy-disease-classification"

CANDIDATE_PATHS = [
    Path("/kaggle/input") / comp,
    Path("/kaggle/input") / "paddy-disease-classification",
    Path("/kaggle/data") / comp,
    Path("/kaggle/data") / "paddy-disease-classification",
    Path("/kaggle/working") / comp,
    Path("/kaggle/working") / "paddy-disease-classification",
]


def _looks_like_dataset_root(p: Path) -> bool:
    return (p / "train.csv").exists() and (p / "sample_submission.csv").exists()


def _find_dataset_root() -> Path:
    for p in CANDIDATE_PATHS:
        if p.exists() and _looks_like_dataset_root(p):
            return p
        if p.exists() and (p / comp).exists() and _looks_like_dataset_root(p / comp):
            return p / comp
    for p in CANDIDATE_PATHS:
        if p.exists():
            for child in p.iterdir():
                if child.is_dir() and _looks_like_dataset_root(child):
                    return child
                if (
                    child.is_dir()
                    and (child / comp).exists()
                    and _looks_like_dataset_root(child / comp)
                ):
                    return child / comp
    raise FileNotFoundError(
        f"Could not locate dataset folder. Tried: {CANDIDATE_PATHS}"
    )


path = _find_dataset_root()
path



## === cell 1
(
    list(path.iterdir())[:20],
    (path / "train_images").exists(),
    (path / "test_images").exists(),
    (path / "train.csv").exists(),
    (path / "sample_submission.csv").exists(),
    "fastai_available=" + str(FASTAI_AVAILABLE),
)



## === cell 2
ss = pd.read_csv(path / "sample_submission.csv")
ss.head(), ss.shape



## === cell 3
FALLBACK_LABEL = "normal"

out_path = Path("submission.csv")


def write_fallback_submission(reason: str):
    sub = ss.copy()
    sub["label"] = FALLBACK_LABEL
    sub.to_csv(out_path, index=False)
    print("Wrote fallback submission due to:", reason)
    print("Wrote:", out_path.resolve())
    print(sub.head())


if not FASTAI_AVAILABLE:
    write_fallback_submission(f"fastai/timm import failed: {FASTAI_IMPORT_ERROR}")



## === cell 4
if not FASTAI_AVAILABLE:
    raise SystemExit(0)

trn_path = path / "train_images"
tst_path = path / "test_images"

assert trn_path.exists(), f"Missing train_images at {trn_path}"
assert tst_path.exists(), f"Missing test_images at {tst_path}"

(trn_path.exists(), tst_path.exists())



## === cell 5
_cpu = os.cpu_count() or 2
num_workers = min(4, _cpu)

dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(128, method="pad", pad_mode="zeros"),
    batch_tfms=aug_transforms(size=128, min_scale=0.75),
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
)

dls = dls.new(
    item_tfms=dls.train.after_item, after_item=dls.train.after_item, cache=True
)

dls



## === cell 6
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")
learn = learn.to_fp16()
learn



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4093470189.py in <cell line: 0>()
      1 # Keep identical model/architecture and training approach.
----> 2 learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".")
      3 learn = learn.to_fp16()
      4 learn
      5 

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in vision_learner(dls, arch, normalize, n_out, pretrained, weights, loss_func, opt_func, lr, splitter, cbs, metrics, path, model_dir, wd, wd_bn_bias, train_bn, moms, cut, init, custom_head, concat_pool, pool, lin_ftrs, ps, first_bn, bn_final, lin_first, y_range, **kwargs)
    233     if isinstance(arch, str):
    234         model,cfg = create_timm_model(arch, n_out, default_split, pretrained, **model_args)
--> 235         if normalize: _timm_norm(dls, cfg, pretrained, n_in)
    236     else:
    237         if normalize: _add_norm(dls, meta, pretrained, n_in)

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in _timm_norm(dls, cfg, pretrained, n_in)
    213     if not dls.after_batch.fs.filter(risinstance(Normalize)):
    214         tfm = Normalize.from_stats(cfg['mean'],cfg['std'])
--> 215         dls.add_tfms([tfm],'after_batch')
    216 
    217 # %% ../../nbs/21_vision.learner.ipynb 42

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

AttributeError: add_tfms

## === cell 7
None



## === cell 8
learn.fine_tune(3, 0.01)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/222546778.py in <cell line: 0>()
      1 # Training logic unchanged.
----> 2 learn.fine_tune(3, 0.01)
      3 

NameError: name 'learn' is not defined

## === cell 9
tst_ids = ss["image_id"].to_numpy()
tst_files = [tst_path / x for x in tst_ids]

tst_dl = dls.test_dl(
    tst_files, with_labels=False, num_workers=num_workers, pin_memory=True
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1796291113.py in <cell line: 0>()
      3 tst_files = [tst_path / x for x in tst_ids]
      4 
----> 5 tst_dl = dls.test_dl(
      6     tst_files, with_labels=False, num_workers=num_workers, pin_memory=True
      7 )

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

AttributeError: test_dl

## === cell 10
import torch

with torch.inference_mode():
    probs, _ = learn.get_preds(dl=tst_dl)

idxs = probs.argmax(dim=1)
probs.shape, idxs.shape



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771573865.py in <cell line: 0>()
      2 
      3 with torch.inference_mode():
----> 4     probs, _ = learn.get_preds(dl=tst_dl)
      5 
      6 idxs = probs.argmax(dim=1)

NameError: name 'learn' is not defined

## === cell 11
vocab = np.array(list(dls.vocab), dtype=object)
idxs_np = idxs.cpu().numpy()
results = pd.Series(vocab[idxs_np])

results.head(), results.nunique(), sorted(results.unique())[:10]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2302327870.py in <cell line: 0>()
      2 # Correctness preserved: same argmax indices mapped to the same dls.vocab strings.
      3 vocab = np.array(list(dls.vocab), dtype=object)
----> 4 idxs_np = idxs.cpu().numpy()
      5 results = pd.Series(vocab[idxs_np])
      6 

NameError: name 'idxs' is not defined

## === cell 12
ss_out = ss.copy()
ss_out["label"] = results.values
ss_out.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print(ss_out.head())
print("CSV preview:")
with open(out_path, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/898779009.py in <cell line: 0>()
      1 ss_out = ss.copy()
----> 2 ss_out["label"] = results.values
      3 ss_out.to_csv(out_path, index=False)
      4 
      5 print("Wrote:", out_path.resolve())

NameError: name 'results' is not defined
