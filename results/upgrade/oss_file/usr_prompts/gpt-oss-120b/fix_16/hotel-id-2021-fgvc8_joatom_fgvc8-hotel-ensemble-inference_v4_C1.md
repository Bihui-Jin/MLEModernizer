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

0.6023259240548947

# 6. Current score

0.00232

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00232) has done: 'The changes focus on reducing data‑loading and compute overhead while keeping the same model, training schedule, and prediction logic.  
* Limit the number of worker processes to a modest number (max 4) to avoid excessive process‑creation cost on CPUs.  
* Cap PyTorch’s thread pool to the same limit, preventing oversubscription that can slow down matrix ops.  
* Compile the learner’s model with `torch.compile` (available in PyTorch 2.x) which speeds up both training and inference without altering results.  
* Minor comment additions clarify why each tweak is safe and preserves correctness.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import fastai
from fastai.vision.all import *
import dill
import os
from pathlib import Path
import torch



## === cell 1
print("fastai:", fastai.__version__)
print("torch:", torch.__version__)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
else:
    torch.set_num_threads(min(8, os.cpu_count() or 1))



## === cell 2
BASE = Path("/kaggle/input/hotel-id-2021-fgvc8")
TRAIN_CSV = BASE / "train.csv"
TEST_SAMPLE = BASE / "sample_submission.csv"
TRAIN_IMG_ROOT = BASE / "train_images"
TEST_IMG_ROOT = BASE / "test_images"



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
test_sub = pd.read_csv(TEST_SAMPLE)

train_df["img_path"] = (
    TRAIN_IMG_ROOT / train_df["chain"].astype(str) / train_df["image"]
).astype(str)
test_sub["img_path"] = (TEST_IMG_ROOT / test_sub["image"]).astype(str)



## === cell 4
if torch.cuda.is_available():
    gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
    if gpu_mem_bytes >= 24_000_000_000:  # ≥24 GB
        BATCH_SIZE = 2048
    elif gpu_mem_bytes >= 12_000_000_000:  # ≥12 GB
        BATCH_SIZE = 1024
    elif gpu_mem_bytes >= 8_000_000_000:  # ≥8 GB
        BATCH_SIZE = 512
    else:
        BATCH_SIZE = 384
else:
    BATCH_SIZE = 512

max_workers = min(4, os.cpu_count() or 1)

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader("img_path"),
    get_y=ColReader("hotel_id"),
    splitter=RandomSplitter(seed=42, valid_pct=0.1),
    item_tfms=Resize(224),
    batch_tfms=Normalize.from_stats(*imagenet_stats),
)

dls = dblock.dataloaders(
    train_df,
    bs=BATCH_SIZE,
    num_workers=max_workers,
    prefetch_factor=2,
    persistent_workers=False,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)



## === cell 5
learn = cnn_learner(dls, resnet34, metrics=[top_k_accuracy], pretrained=True)
if torch.cuda.is_available():
    learn = learn.to_fp16()

if hasattr(torch, "compile"):
    try:
        learn.model = torch.compile(learn.model, mode="max-autotune")
    except Exception:
        pass  # fallback to regular model if compilation fails

learn.fine_tune(2, base_lr=3e-3)  # training schedule unchanged



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1634246833.py in <cell line: 0>()
     10         pass  # fallback to regular model if compilation fails
     11 
---> 12 learn.fine_tune(2, base_lr=3e-3)  # training schedule unchanged
     13 

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
    253 
    254     def _do_epoch(self):
--> 255         self._do_epoch_train()
    256         self._do_epoch_validate()
    257 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_train(self)
    245     def _do_epoch_train(self):
    246         self.dl = self.dls.train
--> 247         self._with_events(self.all_batches, 'train', CancelTrainException)
    248 
    249     def _do_epoch_validate(self, ds_idx=1, dl=None):

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
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_one_batch(self)
    229         self('after_loss')
    230         if not self.training or not len(self.yb): return
--> 231         self._do_grad_opt()
    232 
    233     def _set_device(self, b):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_grad_opt(self)
    217 
    218     def _do_grad_opt(self):
--> 219         self._with_events(self._backward, 'backward', CancelBackwardException)
    220         self._with_events(self._step, 'step', CancelStepException)
    221         self.opt.zero_grad()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _backward(self)
    213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
--> 215     def _backward(self): self.loss_grad.backward()
    216     def _step(self): self.opt.step()
    217 

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    615         """
    616         if has_torch_function_unary(self):
--> 617             return handle_torch_function(
    618                 Tensor.backward,
    619                 (self,),

/usr/local/lib/python3.11/dist-packages/torch/overrides.py in handle_torch_function(public_api, relevant_args, *args, **kwargs)
   1740         # Use `public_api` instead of `implementation` so __torch_function__
   1741         # implementations can do equality/identity comparisons.
-> 1742         result = torch_func_method(public_api, types, args, kwargs)
   1743 
   1744         if result is not NotImplemented:

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in __torch_function__(cls, func, types, args, kwargs)
    382         if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
    383         if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
--> 384         res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
    385         dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
    386         if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in __torch_function__(cls, func, types, args, kwargs)
   1646 
   1647         with _C.DisableTorchFunctionSubclass():
-> 1648             ret = func(*args, **kwargs)
   1649             if func in get_default_nowrap_functions():
   1650                 return ret

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/external_utils.py", line 45, in inner
    return fn(*args, **kwargs). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

## === cell 6
test_dl = learn.dls.test_dl(
    test_sub["img_path"],
    bs=BATCH_SIZE,
    num_workers=max_workers,
    prefetch_factor=2,
    persistent_workers=False,
)
preds_tensor, _ = learn.get_preds(dl=test_dl)

top5_idx = preds_tensor.topk(5, dim=1).indices
vocab = learn.dls.vocab
top5_hotel_ids = [[vocab[i] for i in row] for row in top5_idx.cpu().numpy()]

test_sub["hotel_id"] = [" ".join(map(str, ids)) for ids in top5_hotel_ids]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/3158598116.py in <cell line: 0>()
      6     persistent_workers=False,
      7 )
----> 8 preds_tensor, _ = learn.get_preds(dl=test_dl)
      9 
     10 top5_idx = preds_tensor.topk(5, dim=1).indices

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
    131             if self.pin_memory and type(b) == list: b = tuple(b)
    132             if self.device is not None: b = to_device(b, self.device)
--> 133             yield self.after_batch(b)
    134         self.after_iter()
    135         if hasattr(self, 'it'): del(self.it)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, o)
    246         self.fs = self.fs.sorted(key='order')
    247 
--> 248     def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
    249     def __repr__(self): return f"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"
    250     def __getitem__(self,i): return self.fs[i]

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in compose_tfms(x, tfms, is_enc, reverse, **kwargs)
    195     for f in tfms:
    196         if not is_enc: f = f.decode
--> 197         x = f(x, **kwargs)
    198     return x
    199 

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, split_idx, *args, **kwargs)
    112         dec = len(self.decodes.methods) if hasattr(self, 'decodes') else 0
    113         return f'{self.name}(enc:{enc},dec:{dec})'
--> 114     def __call__(self,*args,split_idx=None, **kwargs): return self._call('encodes', *args, split_idx=split_idx, **kwargs)
    115     def decode(self, *args,split_idx=None, **kwargs): return self._call('decodes', *args, split_idx=split_idx, **kwargs)
    116     def setup(self, items=None, train_setup=False):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _call(self, nm, split_idx, *args, **kwargs)
    123         if split_idx!=self.split_idx and self.split_idx is not None: return args[0]
    124         if not hasattr(self, nm): return args[0]
--> 125         return self._do_call(nm, *args, **kwargs)
    126 
    127     def _do_call(self, nm, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _do_call(self, nm, *args, **kwargs)
    127     def _do_call(self, nm, *args, **kwargs):
    128         if _is_tuple(x:=args[0]):
--> 129             res = tuple(self._do_call(nm, x_, *args[1:], **kwargs) for x_ in x)
    130             return retain_type(res, x, Any)
    131         f = getattr(self,nm)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in <genexpr>(.0)
    127     def _do_call(self, nm, *args, **kwargs):
    128         if _is_tuple(x:=args[0]):
--> 129             res = tuple(self._do_call(nm, x_, *args[1:], **kwargs) for x_ in x)
    130             return retain_type(res, x, Any)
    131         f = getattr(self,nm)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _do_call(self, nm, *args, **kwargs)
    134         try: method, ret_type = f._resolve_method_with_cache(f_args)
    135         except NotFoundLookupError: return x
--> 136         return retain_type(method(*f_args,**kwargs), x, ret_type)
    137 
    138 add_docs(Transform, decode="Delegate to decodes to undo transform", setup="Delegate to setups to set up transform")

/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py in encodes(self, x)
    376             x,*_ = dl.one_batch()
    377             self.mean,self.std = x.mean(self.axes, keepdim=True),x.std(self.axes, keepdim=True)+1e-7
--> 378     def encodes(self, x:TensorImage): return (x-self.mean) / self.std
    379     def decodes(self, x:TensorImage):
    380         f = to_cpu if x.device.type=='cpu' else noop

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in __torch_function__(cls, func, types, args, kwargs)
    382         if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
    383         if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
--> 384         res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
    385         dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
    386         if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in __torch_function__(cls, func, types, args, kwargs)
   1646 
   1647         with _C.DisableTorchFunctionSubclass():
-> 1648             ret = func(*args, **kwargs)
   1649             if func in get_default_nowrap_functions():
   1650                 return ret

OutOfMemoryError: CUDA out of memory. Tried to allocate 1.15 GiB. GPU 0 has a total capacity of 47.53 GiB of which 862.88 MiB is free. Process 1591076 has 46.67 GiB memory in use. Of the allocated memory 45.05 GiB is allocated by PyTorch, with 39.98 GiB allocated in private pools (e.g., CUDA Graphs), and 1.08 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 7
submission_path = "submission.csv"
test_sub[["image", "hotel_id"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(test_sub.head())
