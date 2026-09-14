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

# 5. Target score

0.616496169066261

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00092) has done: 'The fix completes the unfinished conditional, correctly builds the top‑5 prediction strings, and ensures a CSV file named `submission.csv` is written. Minor adjustments include converting image paths to strings, handling the case where no model loads (fallback to the most common hotel), and using `torch.topk` on the accumulated probability tensor. The core model‑loading and ensembling logic remains unchanged.'
- What this solution (achieved 0.00209) has done: 'I adjust the inference cell so that when no model files are found (or they all fail to load) the fallback prediction uses the five most frequent hotels in the training data instead of repeating a single ID, which should raise MAP@5. Additionally, when any models do load I average the accumulated probabilities across models before extracting the top‑5 predictions, a small change that keeps the original ensemble logic while improving calibration. These tweaks keep the core workflow intact and aim to move the score much closer to the target.'
- What this solution (achieved 0.00209) has done: 'I adjust the model file paths to the proper absolute location under `/kaggle/input/…` so the pre‑trained learners can be loaded, average the probabilities whenever at least one model is loaded, and keep the fallback to the five most frequent hotels unchanged. These minimal fixes let the original ensemble logic run and should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I add a simple, data‑driven fallback: when none of the pre‑trained models can be loaded, the script predict the five most frequent hotels *per chain* (derived from the test image’s parent folder) instead of a single global list. This keeps the original workflow intact, only improves the “no‑model” case, and should raise MAP@5 toward the target without altering the core model logic.'
- What this solution (achieved 0.00209) has done: 'I keep the original workflow intact and only adjust the fallback prediction logic. When no pretrained model can be loaded, the code now builds a combined top‑5 list: it starts with the five most frequent hotels for the image’s chain (if known) and fills any remaining slots with the global most‑frequent hotels, avoiding duplicates. This modest change increases the chance that the correct hotel appears in the submitted list, moving the MAP@5 score closer to the target while preserving all core logic.'
- What this solution (achieved 0.00209) has done: 'I add a small discovery step that automatically finds any exported FastAI‑learner `.pkl` files under the `/kaggle/input` directory, so the script can actually load the pretrained models that are present in the environment instead of relying on hard‑coded (non‑existent) paths. This modest change lets the original ensemble logic run when models are found, which should raise MAP@5 toward the target while keeping all core logic unchanged. If no models are discovered the existing fallback (chain‑wise top‑5 plus global top‑5) remains unchanged.'
- What this solution (achieved 0.00209) has done: 'I expanded the model discovery to look for **any** FastAI learner `.pkl` files under both `/kaggle/input` and `/kaggle/working` (instead of only `export_*.pkl`). This lets the script actually load pretrained models that may be present, while keeping the original ensemble / fallback workflow unchanged, so the MAP@5 score should move closer to the target.'
- What this solution (achieved 0.00209) has done: 'I adjust the inference loop so that the test dataloader receives the list of image file paths (instead of the whole DataFrame) and use `learn.get_preds` for a straightforward probability extraction. This small fix lets any discovered FastAI models actually produce predictions, which should raise the MAP@5 score toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.00209) has done: 'I guard the fallback path so that when no pretrained models are found (or the quick‑training data isn’t available) the script skips the costly ImageDataLoaders step and directly builds predictions using the chain‑wise and global top‑5 hotel lists. This removes the FileNotFoundError, guarantees `preds` is always defined, and keeps the core logic unchanged while modestly improving MAP@5 via chain‑aware fallback predictions.'
- What this solution (achieved 0.00209) has done: 'I keep the overall workflow unchanged but fix two subtle issues that can increase MAP@5: (1) ensure model files are reliably found by searching both the `/kaggle/input` and `/kaggle/working` directories for any `.pkl` learner files, (2) after obtaining the top‑5 predictions from the model(s), guarantee that the list contains five **unique** hotel IDs – if duplicates appear we fill the gaps with the chain‑specific or global most‑frequent hotels. This small correctness tweak preserves the original ensemble logic while improving the chance that the true hotel appears in the submitted list, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import torch
import fastai
from fastai.vision.all import *
import dill
import warnings
from pathlib import Path
from collections import Counter



## === cell 1
print("fastai", fastai.__version__, "torch", torch.__version__)



## === cell 2
model_paths = list(Path("/kaggle/input").rglob("*.pkl")) + list(
    Path("/kaggle/working").rglob("*.pkl")
)
if not model_paths:
    warnings.warn(
        "No *.pkl model files found under /kaggle/input or /kaggle/working; will train a quick model and use fallback predictions."
    )
else:
    print(f"Discovered {len(model_paths)} model file(s):")
    for mp in model_paths:
        print(f"  - {mp}")

image_path = Path("/kaggle/input/hotel-id-2021-fgvc8/train_images/")
test_image_path = Path("/kaggle/input/hotel-id-2021-fgvc8/test_images/")

submission = pd.read_csv("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
test = submission.copy()
test["image"] = (test_image_path / test["image"]).astype(str)

train_df = pd.read_csv("/kaggle/input/hotel-id-2021-fgvc8/train.csv")

global_top5 = [str(h) for h, _ in Counter(train_df["hotel_id"]).most_common(5)]

chain_top5 = {}
for chain_val, grp in train_df.groupby("chain"):
    chain_top5[int(chain_val)] = [
        str(h) for h, _ in Counter(grp["hotel_id"]).most_common(5)
    ]

probs = None
learn = None  # keep last loaded learner for vocab reference
model_cnt = 0  # count of successfully evaluated models

if not model_paths:
    sample_df = train_df.sample(frac=0.10, random_state=42).reset_index(drop=True)

    sample_df["rel_path"] = (
        sample_df["chain"].astype(str) + "/" + sample_df["image"].astype(str)
    )

    dls = ImageDataLoaders.from_df(
        sample_df,
        path=image_path,
        fn_col="rel_path",  # points to chain sub‑folder + file name
        label_col="hotel_id",
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(128),
        batch_tfms=aug_transforms(size=128),
        bs=64,
    )

    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(1)

    test_dl = learn.dls.test_dl(list(test["image"]))
    probs, _ = learn.get_preds(dl=test_dl)
    model_cnt = 1  # treat this as a single loaded model

else:
    for model_path in model_paths:
        p = Path(model_path)
        if not p.exists():
            warnings.warn(f"Model file not found, skipping: {model_path}")
            continue
        try:
            learn = load_learner(fname=p, cpu=True)
            test_dl = learn.dls.test_dl(list(test["image"]))
            probs_temp, _ = learn.get_preds(dl=test_dl)
            if probs is None:
                probs = probs_temp
            else:
                probs += probs_temp
            model_cnt += 1
        except Exception as e:
            warnings.warn(f"Failed to load or evaluate model {model_path}: {e}")

    if probs is not None and model_cnt >= 1:
        probs = probs / model_cnt  # average probabilities across loaded models

if probs is None:
    preds = []
    for img_path in test["image"]:
        p = Path(img_path)
        try:
            chain_id = int(p.parent.name)
        except Exception:
            chain_id = None
        if chain_id is not None and chain_id in chain_top5:
            top5 = chain_top5[chain_id].copy()
            for gid in global_top5:
                if len(top5) >= 5:
                    break
                if gid not in top5:
                    top5.append(gid)
        else:
            top5 = global_top5.copy()
        preds.append(" ".join(top5))
else:
    top5_idxs = probs.topk(5, dim=1)[1]  # (n_samples, 5) indices
    preds = []
    for i, idx_row in enumerate(top5_idxs):
        hotel_ids = [learn.dls.vocab[int(idx)] for idx in idx_row]
        if len(set(hotel_ids)) < 5:
            img_path = test["image"].iloc[i]
            p = Path(img_path)
            try:
                chain_id = int(p.parent.name)
            except Exception:
                chain_id = None
            filler = chain_top5.get(chain_id, global_top5)
            for gid in filler:
                if len(hotel_ids) >= 5:
                    break
                if gid not in hotel_ids:
                    hotel_ids.append(gid)
            for gid in global_top5:
                if len(hotel_ids) >= 5:
                    break
                if gid not in hotel_ids:
                    hotel_ids.append(gid)
        preds.append(" ".join(map(str, hotel_ids)))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1257410954.py in <cell line: 0>()
     63 
     64     learn = vision_learner(dls, resnet34, metrics=accuracy)
---> 65     learn.fine_tune(1)
     66 
     67     test_dl = learn.dls.test_dl(list(test["image"]))

/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py in fine_tune(self, epochs, base_lr, freeze_epochs, lr_mult, pct_start, div, **kwargs)
    165     "Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR."
    166     self.freeze()
--> 167     self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    168     base_lr /= 2
    169     self.unfreeze()

/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
    119     scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
    120               'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
--> 121     self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)
    122 
    123 # %% ../../nbs/14_callback.schedule.ipynb 50

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
    270             self.opt.set_hypers(lr=self.lr if lr is None else lr)
    271             self.n_epoch = n_epoch
--> 272             self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    273 
    274     def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_fit(self)
    259         for epoch in range(self.n_epoch):
    260             self.epoch=epoch
--> 261             self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    262 
    263     def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch(self)
    254     def _do_epoch(self):
    255         self._do_epoch_train()
--> 256         self._do_epoch_validate()
    257 
    258     def _do_fit(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    250         if dl is None: dl = self.dls[ds_idx]
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 
    254     def _do_epoch(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in __iter__(self)
    127         self.before_iter()
    128         self.__idxs=self.get_idxs() # called in context of main process (not workers/subprocesses)
--> 129         for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
    130             # pin_memory causes tuples to be converted to lists, so convert them back to tuples
    131             if self.pin_memory and type(b) == list: b = tuple(b)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py", line 263, in encodes
    return TensorCategory(self.vocab.o2i[o])
                          ~~~~~~~~~~~~~~^^^
KeyError: 312

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 42, in fetch
    data = next(self.dataset_iter)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 140, in create_batches
    yield from map(self.do_batch, self.chunkify(res))
  File "/usr/local/lib/python3.11/dist-packages/fastcore/basics.py", line 265, in chunked
    res = list(itertools.islice(it, chunk_sz))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 170, in do_item
    try: return self.after_item(self.create_item(s))
                                ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 177, in create_item
    if self.indexed: return self.dataset[s or 0]
                            ~~~~~~~~~~~~^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 454, in __getitem__
    res = tuple([tl[it] for tl in self.tls])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 454, in <listcomp>
    res = tuple([tl[it] for tl in self.tls])
                 ~~^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 413, in __getitem__
    return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/core.py", line 373, in _after_item
    def _after_item(self, o): return self.tfms(o)
                                     ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 248, in __call__
    def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 197, in compose_tfms
    x = f(x, **kwargs)
        ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 114, in __call__
    def __call__(self,*args,split_idx=None, **kwargs): return self._call('encodes', *args, split_idx=split_idx, **kwargs)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 125, in _call
    return self._do_call(nm, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py", line 136, in _do_call
    return retain_type(method(*f_args,**kwargs), x, ret_type)
                       ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py", line 265, in encodes
    raise KeyError(f"Label '{o}' was not included in the training dataset") from e
KeyError: "Label '312' was not included in the training dataset"


## === cell 3
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
submission.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4177152164.py in <cell line: 0>()
      1 # Write the submission file
----> 2 submission["hotel_id"] = preds
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission written to submission.csv")
      5 submission.head()

NameError: name 'preds' is not defined
