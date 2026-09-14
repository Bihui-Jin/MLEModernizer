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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.03227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pathlib import Path
import zipfile, csv, gc
import torch, torch.nn as nn
from fastai.vision.all import *
from skimage.color import rgb2gray as _rgb2gray



## === cell 1
path_root = Path("/kaggle/input/denoising-dirty-documents")
if not path_root.exists():
    path_root = Path(".")
path_train = path_root / "train"
path_train_clean = path_root / "train_cleaned"
path_test = path_root / "test"
path_submission = Path("submission")
path_submission.mkdir(exist_ok=True)




## === cell 2
def get_target(fn: Path) -> Path:
    return path_train_clean / fn.name


denoise_block = DataBlock(
    blocks=(ImageBlock, ImageBlock(cls=PILImageBW)),  # input RGB, target grayscale
    get_items=get_image_files,
    get_y=get_target,
    splitter=RandomSplitter(seed=42),
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(mult=1.0),
)

dls = denoise_block.dataloaders(path_train, bs=4, num_workers=0)



## === cell 3
learn = unet_learner(dls, resnet34, loss_func=nn.L1Loss(), metrics=rmse, n_out=1)
learn.model_dir = Path("models")
learn.freeze()
learn.fit_one_cycle(5, 1e-3)  # quick training
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5, 1e-3))  # fine‑tuning




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/1576578199.py in <cell line: 0>()
      2 learn.model_dir = Path("models")
      3 learn.freeze()
----> 4 learn.fit_one_cycle(5, 1e-3)  # quick training
      5 learn.unfreeze()
      6 learn.fit_one_cycle(5, slice(1e-5, 1e-3))  # fine‑tuning

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

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in one_batch(self, i, b)
    241         b = self._set_device(b)
    242         self._split(b)
--> 243         self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    244 
    245     def _do_epoch_train(self):

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

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in after_batch(self)
    572         if len(self.yb) == 0: return
    573         mets = self._train_mets if self.training else self._valid_mets
--> 574         for met in mets: met.accumulate(self.learn)
    575         if not self.training: return
    576         self.lrs.append(self.opt.hypers[-1]['lr'])

/usr/local/lib/python3.11/dist-packages/fastai/metrics.py in accumulate(self, learn)
     49         elif self.dim_argmax: pred = pred.argmax(dim=self.dim_argmax)
     50         if self.thresh:  pred = (pred >= self.thresh)
---> 51         self.accum_values(pred,learn.y,learn)
     52 
     53     def accum_values(self, preds, targs,learn=None):

/usr/local/lib/python3.11/dist-packages/fastai/metrics.py in accum_values(self, preds, targs, learn)
     55         to_d = learn.to_detach if learn is not None else to_detach
     56         preds,targs = to_d(preds),to_d(targs)
---> 57         if self.flatten: preds,targs = flatten_check(preds,targs)
     58         self.preds.append(preds)
     59         self.targs.append(targs)

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in flatten_check(inp, targ)
    787     "Check that `inp` and `targ` have the same number of elements and flatten them."
    788     inp,targ = TensorBase(inp.contiguous()).view(-1),TensorBase(targ.contiguous()).view(-1)
--> 789     test_eq(len(inp), len(targ))
    790     return inp,targ
    791 

/usr/local/lib/python3.11/dist-packages/fastcore/test.py in test_eq(a, b)
     38 def test_eq(a,b):
     39     "`test` that `a==b`"
---> 40     test(a,b,equals, cname='==')
     41 
     42 # %% ../nbs/00_test.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/test.py in test(a, b, cmp, cname)
     28     "`assert` that `cmp(a,b)`; display inputs and `cname or cmp.__name__` if it fails"
     29     if cname is None: cname=cmp.__name__
---> 30     assert cmp(a,b),f"{cname}:\n{a}\n{b}"
     31 
     32 # %% ../nbs/00_test.ipynb

AssertionError: Exception occured in `Recorder` when calling event `after_batch`:
	==:
65536
196608

## === cell 4
def rgb2gray(tensor):
    """Convert a 3‑channel tensor to 1‑channel grayscale."""
    if tensor.shape[0] == 1:  # already single channel
        return tensor
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    gray = _rgb2gray(np_img)  # returns HxW float32 in [0,1]
    return torch.from_numpy(gray).unsqueeze(0).float()


submission_path = path_submission / "submission.csv"
with submission_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    for img_path in sorted(path_test.glob("*.png")):
        img_id = int(img_path.stem)  # filename without .png
        img = PILImage.create(img_path)
        pred = learn.predict(img)[0]  # tensor (C,H,W)
        pred = rgb2gray(pred).clamp(0, 1)  # ensure single channel in [0,1]
        _, h, w = pred.shape
        for r in range(h):
            for c in range(w):
                pid = f"{img_id}_{r+1}_{c+1}"
                val = pred[0, r, c].item()
                writer.writerow([pid, val])
print(f"Submission written to {submission_path}")
