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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.9648

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The timeout is dominated by training three heavy ImageNet backbones (DenseNet161/201 + ResNet50) and then running full-dataset inference twice (all-train + all-test) for each model. To keep identical core logic while cutting runtime, the main changes are: (1) enable fastai/PyTorch mixed-precision for the vision learners (same architecture/training loop, faster GPU math), (2) avoid redundant CPU overhead in the image pipeline (no extra augmentation work during logits extraction; keep augmentations for training only), (3) speed up dataloading/inference by tuning DataLoader parameters (workers, prefetch, persistent workers, pinning) and using a no-grad/prediction context that preserves outputs, and (4) make caching more reliable so reruns never retrain/infer unnecessarily. The tabular stacker training is left logically identical (same layers/loss/training schedule), but we reduce overhead in the metric computation by accumulating predictions/targets once per epoch instead of repeatedly converting per-batch.'
- What this solution (achieved 0.5) has done: 'I fix the fastai API break that prevents the tabular stacker from being created (the `ps` argument is no longer accepted by `tabular_learner`) by moving dropout probability into `config`, which preserves the same model semantics. I also fix the callback monitor name mismatch by explicitly naming the AUC metric so `SaveModelCallback` can track it, enabling the model to be saved/loaded correctly. Finally, I correct the test inference pipeline so `test_dl` is built from the same `TabularPandas` preprocessing (instead of `dls.test_dl(test)` which is incompatible here) and then write a valid `submission.csv` with aligned ids/predictions; these fixes should move the score up from 0.5 toward the target because your current 0.5 is consistent with broken/constant predictions due to the runtime errors.'

# 9. Code solution

## === cell 0
import os, gc, random, json, time
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from fastai.vision.all import *
from fastai.tabular.all import *

SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

set_seed(SEED, reproducible=True)

try:
    torch.set_num_threads(min(8, os.cpu_count() or 2))
except Exception:
    pass

torch.backends.cudnn.benchmark = True




## === cell 1
DATA_ROOT = Path("../input/histopathologic-cancer-detection")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
LABELS_CSV = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

assert (
    TRAIN_DIR.exists() and TEST_DIR.exists()
), f"Missing train/test dirs under {DATA_ROOT}"
assert LABELS_CSV.exists() and SAMPLE_SUB.exists(), "Missing required CSVs"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["id"] = labels_df["id"].astype(str)
labels_df["label"] = labels_df["label"].astype(int)

label_map = labels_df.set_index("id")["label"].to_dict()

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["id"] = sample_df["id"].astype(str)

print(labels_df.head())
print("Train images (from CSV):", len(labels_df))
print("Test images  (from CSV):", len(sample_df))




## === cell 2
train_files = [TRAIN_DIR / f"{i}.tif" for i in labels_df["id"].values]
train_files = L(train_files)

idx = np.arange(len(train_files))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
valid_pct = 0.2
n_valid = int(len(idx) * valid_pct)
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

train_items = train_files[train_idx]
valid_items = train_files[valid_idx]

len(train_items), len(valid_items)




## === cell 3
def _cache_paths(name: str):
    cache_dir = Path("./logit_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"{name}_train_logits.npy",
        cache_dir / f"{name}_test_logits.npy",
        cache_dir / f"{name}_meta.json",
    )


def _save_logits(name, p_all, p_test, test_ids):
    tr_path, te_path, meta_path = _cache_paths(name)
    np.save(tr_path, p_all)
    np.save(te_path, p_test)
    meta = {
        "name": name,
        "seed": SEED,
        "n_train": int(p_all.shape[0]),
        "n_test": int(p_test.shape[0]),
        "test_ids_first10": test_ids[:10],
        "test_ids_last10": test_ids[-10:],
        "created_utc": time.time(),
        "dtype_train": str(p_all.dtype),
        "dtype_test": str(p_test.dtype),
    }
    meta_path.write_text(json.dumps(meta))
    return tr_path, te_path


def _load_logits(name, expected_n_train, expected_test_ids):
    tr_path, te_path, meta_path = _cache_paths(name)
    if not (tr_path.exists() and te_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("seed") != SEED:
        return None
    p_all = np.load(tr_path, mmap_mode="r")
    p_test = np.load(te_path, mmap_mode="r")
    if p_all.shape[0] != expected_n_train:
        return None
    if p_test.shape[0] != len(expected_test_ids):
        return None
    if meta.get("test_ids_first10") != expected_test_ids[:10]:
        return None
    if meta.get("test_ids_last10") != expected_test_ids[-10:]:
        return None
    return np.asarray(p_all), np.asarray(p_test)


def _ensure_2col_logits(p):
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.shape[1] == 1:
        p = np.concatenate([np.zeros((p.shape[0], 1), dtype=p.dtype), p], axis=1)
    return p


@torch.inference_mode()
def get_test_logits(learn, test_files, bs=256):
    test_dl = learn.dls.test_dl(test_files, with_labels=False, bs=bs, shuffle=False)
    p_test, _ = learn.get_preds(dl=test_dl, act=None)
    return p_test.cpu().numpy()


vision_cfg = [
    ("dense161", densenet161),
    ("dense201", densenet201),
    ("res50", resnet50),
]

vision_logits = {}
test_ids = sample_df["id"].tolist()
test_files = [TEST_DIR / f"{i}.tif" for i in test_ids]
test_files = L(test_files)
test_files_ref = test_files

splitter = IndexSplitter(list(valid_idx))

cpu_cnt = os.cpu_count() or 2
nw_vision = min(6, cpu_cnt)

for name, arch in vision_cfg:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cached = _load_logits(
        name, expected_n_train=len(train_files), expected_test_ids=test_ids
    )
    if cached is not None:
        p_all, p_test = cached
        vision_logits[name] = {
            "train": _ensure_2col_logits(p_all),
            "test": _ensure_2col_logits(p_test),
        }
        continue

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=lambda o: o,
        get_y=lambda o: label_map[o.stem],
        splitter=splitter,
        item_tfms=Resize(96),
        batch_tfms=[
            *aug_transforms(size=96, min_scale=0.9),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    dls = dblock.dataloaders(
        train_files,
        bs=128,
        shuffle=True,
        num_workers=nw_vision,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw_vision > 0),
        prefetch_factor=2 if nw_vision > 0 else None,
    )

    learn = vision_learner(dls, arch, metrics=[RocAucBinary()], pretrained=True)

    if torch.cuda.is_available():
        learn = learn.to_fp16()

    learn.fit_one_cycle(1, 3e-3)

    @torch.inference_mode()
    def _get_logits_for_files(files, with_labels, bs):
        dl = dls.test_dl(files, with_labels=with_labels, bs=bs, shuffle=False)
        p, t = learn.get_preds(dl=dl, act=None)
        return p.cpu().numpy(), (t.cpu().numpy() if t is not None else None)

    p_all, _ = _get_logits_for_files(train_files, with_labels=True, bs=512)
    p_all = _ensure_2col_logits(p_all)

    p_test = _ensure_2col_logits(get_test_logits(learn, test_files, bs=512))

    vision_logits[name] = {"train": p_all, "test": p_test}
    _save_logits(name, p_all, p_test, test_ids)

{k: (v["train"].shape, v["test"].shape) for k, v in vision_logits.items()}




## === cell 4
train = pd.DataFrame(
    {
        "dense161_0": vision_logits["dense161"]["train"][:, 0],
        "dense161_1": vision_logits["dense161"]["train"][:, 1],
        "dense201_0": vision_logits["dense201"]["train"][:, 0],
        "dense201_1": vision_logits["dense201"]["train"][:, 1],
        "res50_0": vision_logits["res50"]["train"][:, 0],
        "res50_1": vision_logits["res50"]["train"][:, 1],
        "y": labels_df["label"].values.astype(np.int64),
    }
)

test = pd.DataFrame(
    {
        "dense161_0": vision_logits["dense161"]["test"][:, 0],
        "dense161_1": vision_logits["dense161"]["test"][:, 1],
        "dense201_0": vision_logits["dense201"]["test"][:, 0],
        "dense201_1": vision_logits["dense201"]["test"][:, 1],
        "res50_0": vision_logits["res50"]["test"][:, 0],
        "res50_1": vision_logits["res50"]["test"][:, 1],
    }
)
test["y"] = 0

train.head(), test.head()




## === cell 5
dep_var = "y"
cont_names = [
    "dense161_0",
    "dense161_1",
    "dense201_0",
    "dense201_1",
    "res50_0",
    "res50_1",
]

splits = RandomSplitter(valid_pct=0.2, seed=SEED)(range_of(train))

to = TabularPandas(
    train,
    procs=[Normalize],
    cont_names=cont_names,
    y_names=dep_var,
    splits=splits,
    y_block=CategoryBlock,
)

nw = min(4, os.cpu_count() or 2)
dls = to.dataloaders(
    bs=512,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
)

test_to = to.new(test)
test_dl = test_to.dataloaders(
    bs=512,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
).train




## === cell 6
def _auc_accum(inp, targ):
    if inp.ndim == 1 or inp.shape[1] == 1:
        prob1 = torch.sigmoid(inp.view(-1))
    else:
        prob1 = torch.softmax(inp, dim=1)[:, 1]
    targ = targ.view(-1).long()
    return roc_auc_score(targ.cpu().numpy(), prob1.detach().cpu().numpy())


roc_auc_metric = AccumMetric(_auc_accum, flatten=False)
roc_auc_metric.name = "roc_auc"




## === cell 7
tab_config = tabular_config(ps=0.5)

learn = tabular_learner(
    dls,
    layers=[10, 10, 10],
    metrics=[accuracy, roc_auc_metric],
    config=tab_config,
    wd=1e-1,
)

cbs = [
    SaveModelCallback(monitor="roc_auc", fname="best", with_opt=True),
]
learn.fit_one_cycle(20, 1e-3, cbs=cbs)




## === cell 8
learn.load("best")




## === cell 9
preds, _ = learn.get_preds(dl=test_dl)
preds = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()

val_res = learn.validate()
metric_names = learn.recorder.metric_names
print("Metric names:", metric_names)

auc_val = None
if "roc_auc" in metric_names:
    auc_val = float(
        val_res[metric_names.index("roc_auc") - 1]
    )  # validate() excludes 'epoch'
else:
    auc_val = float(val_res[-1])

auc_val




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1045200646.py in <cell line: 0>()
----> 1 preds, _ = learn.get_preds(dl=test_dl)
      2 preds = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()
      3 
      4 val_res = learn.validate()
      5 # recorder.metric_names typically: ['epoch','train_loss','valid_loss','accuracy','roc_auc']

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in get_preds(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)
    314         if with_loss: ctx_mgrs.append(self.loss_not_reduced())
    315         with ContextManagers(ctx_mgrs):
--> 316             self._do_epoch_validate(dl=dl)
    317             if act is None: act = getcallable(self.loss_func, 'activation')
    318             res = cb.all_tensors()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    250         if dl is None: dl = self.dls[ds_idx]
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 
    254     def _do_epoch(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
--> 209         self(f'after_{event_type}');  final()
    210 
    211     def all_batches(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in __call__(self, event_name)
    178 
    179     def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
--> 180     def __call__(self, event_name): L(event_name).map(self._call_one)
    181 
    182     def _call_one(self, event_name):

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    166     def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    170     def argfirst(self, f, negate=False):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    949     res = map(g, iterable)
    950     if gen: return res
--> 951     return list(res)
    952 
    953 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    934             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 
    938 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _call_one(self, event_name)
    182     def _call_one(self, event_name):
    183         if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
--> 184         for cb in self.cbs.sorted('order'): cb(event_name)
    185 
    186     def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
---> 64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)
     65         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
     66         return res

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     60         res = None
     61         if self.run and _run:
---> 62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
     64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in after_validate(self)
    587     def before_validate(self): self._valid_mets.map(Self.reset())
    588     def after_train   (self): self.log += self._train_mets.map(_maybe_item)
--> 589     def after_validate(self): self.log += self._valid_mets.map(_maybe_item)
    590     def after_cancel_train(self):    self.cancel_train = True
    591     def after_cancel_validate(self): self.cancel_valid = True

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    166     def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    170     def argfirst(self, f, negate=False):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    949     res = map(g, iterable)
    950     if gen: return res
--> 951     return list(res)
    952 
    953 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    934             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 
    938 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _maybe_item(t)
    541 # %% ../nbs/13a_learner.ipynb 134
    542 def _maybe_item(t):
--> 543     t = t.value
    544     try: return t.item()
    545     except: return t

/usr/local/lib/python3.11/dist-packages/fastai/metrics.py in value(self)
     71         preds,targs = torch.cat(self.preds),torch.cat(self.targs)
     72         if self.to_np: preds,targs = preds.numpy(),targs.numpy()
---> 73         return self.func(targs, preds, **self.kwargs) if self.invert_args else self.func(preds, targs, **self.kwargs)
     74 
     75     @property

/tmp/ipykernel_11/3874750844.py in _auc_accum(inp, targ)
      5         prob1 = torch.softmax(inp, dim=1)[:, 1]
      6     targ = targ.view(-1).long()
----> 7     return roc_auc_score(targ.cpu().numpy(), prob1.detach().cpu().numpy())
      8 
      9 

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_ranking.py in roc_auc_score(y_true, y_score, average, sample_weight, max_fpr, multi_class, labels)
    570         labels = np.unique(y_true)
    571         y_true = label_binarize(y_true, classes=labels)[:, 0]
--> 572         return _average_binary_score(
    573             partial(_binary_roc_auc_score, max_fpr=max_fpr),
    574             y_true,

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_base.py in _average_binary_score(binary_metric, y_true, y_score, average, sample_weight)
     73 
     74     if y_type == "binary":
---> 75         return binary_metric(y_true, y_score, sample_weight=sample_weight)
     76 
     77     check_consistent_length(y_true, y_score, sample_weight)

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_ranking.py in _binary_roc_auc_score(y_true, y_score, sample_weight, max_fpr)
    337     """Binary roc auc score."""
    338     if len(np.unique(y_true)) != 2:
--> 339         raise ValueError(
    340             "Only one class present in y_true. ROC AUC score "
    341             "is not defined in that case."

ValueError: Exception occured in `Recorder` when calling event `after_validate`:
	Only one class present in y_true. ROC AUC score is not defined in that case.

## === cell 10
sub = pd.read_csv(SAMPLE_SUB)
sub["id"] = sub["id"].astype(str)

test_ids = [p.stem for p in test_files_ref]
pred_map = pd.DataFrame({"id": test_ids, "label": preds})

sub = sub[["id"]].merge(pred_map, on="id", how="left")
assert (
    sub["label"].notna().all()
), "Some test ids did not receive predictions; ordering/matching issue."

sub.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3327255805.py in <cell line: 0>()
      3 
      4 test_ids = [p.stem for p in test_files_ref]
----> 5 pred_map = pd.DataFrame({"id": test_ids, "label": preds})
      6 
      7 sub = sub[["id"]].merge(pred_map, on="id", how="left")

NameError: name 'preds' is not defined

## === cell 11
out_name = "submission.csv"
sub.to_csv(out_name, index=False)
print("Wrote:", out_name, "rows:", len(sub))
print("Validation AUC (internal stacker split):", auc_val)
print(sub.describe())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1803359758.py in <cell line: 0>()
      2 sub.to_csv(out_name, index=False)
      3 print("Wrote:", out_name, "rows:", len(sub))
----> 4 print("Validation AUC (internal stacker split):", auc_val)
      5 print(sub.describe())

NameError: name 'auc_val' is not defined
