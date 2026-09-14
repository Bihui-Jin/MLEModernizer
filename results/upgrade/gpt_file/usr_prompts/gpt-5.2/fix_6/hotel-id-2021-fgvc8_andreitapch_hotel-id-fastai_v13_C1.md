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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.4475035783447009

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, shutil
from pathlib import Path

shutil.rmtree("./kaggle", ignore_errors=True)
os.makedirs("./kaggle/working", exist_ok=True)



## === cell 1
import warnings
import random

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import fastai
import dill
from fastai.vision.all import *
from fastai.metrics import accuracy, top_k_accuracy



## === cell 2
path = Path("../input/hotel-id-2021-fgvc8")

sample_sub_path = path / "sample_submission.csv"
labels_path = path / "train.csv"

train_images_dir = path / "train_images"
test_images_dir = path / "test_images"

assert sample_sub_path.exists(), f"Missing {sample_sub_path}"
assert labels_path.exists(), f"Missing {labels_path}"
assert train_images_dir.exists(), f"Missing {train_images_dir}"
assert test_images_dir.exists(), f"Missing {test_images_dir}"



## === cell 3
train_data = pd.read_csv(labels_path)
train_data.head(5)



## === cell 4
pass



## === cell 5
train_data["chain"] = train_data["chain"].astype(str)
train_data["image"] = train_data["image"].astype(str)
train_data["image_path"] = (
    train_images_dir.as_posix() + "/" + train_data["chain"] + "/" + train_data["image"]
)
train_data[:5]



## === cell 6
missing = 0
for p in train_data["image_path"].sample(200, random_state=0).tolist():
    if not Path(p).exists():
        missing += 1
assert (
    missing == 0
), f"Found {missing} missing image paths in sample; check directory structure."



## === cell 7
batch_size = 32


def create_dataLoader():
    df = train_data[["image_path", "hotel_id"]].copy()

    item_tfms = None
    batch_tfms = [Resize(224, method="pad", pad_mode="reflection")] + list(
        aug_transforms(size=224)
    )

    dls = ImageDataLoaders.from_df(
        df=df,
        path=".",
        valid_pct=0.2,
        seed=42,
        fn_col="image_path",
        label_col="hotel_id",
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=batch_size,
        num_workers=min(8, os.cpu_count() or 2),
        pin_memory=torch.cuda.is_available(),
    )
    return dls




## === cell 8
dataset = create_dataLoader()
dataset



## === cell 9
pass



## === cell 10
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




## === cell 11
def training_models(dataset, model_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learn = cnn_learner(
            dataset,
            models.resnet101,
            metrics=[accuracy, top_k_accuracy],
            opt_func=QHAdam,
        ).to_fp16()
        learn.fine_tune(5, 0.005, freeze_epochs=3)
        learn.save(model_name)
    return learn




## === cell 12
model_name = "model_allData_rn101"
learn = training_models(dataset, model_name)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1266818602.py in <cell line: 0>()
      1 model_name = "model_allData_rn101"
----> 2 learn = training_models(dataset, model_name)
      3 

/tmp/ipykernel_55/1525030013.py in training_models(dataset, model_name)
      8             opt_func=QHAdam,
      9         ).to_fp16()
---> 10         learn.fine_tune(5, 0.005, freeze_epochs=3)
     11         learn.save(model_name)
     12     return learn

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

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 42, in fetch
    data = next(self.dataset_iter)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 140, in create_batches
    yield from map(self.do_batch, self.chunkify(res))
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 185, in do_batch
    def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 183, in create_batch
    if not self.prebatched: collate_error(e,b)
                            ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 181, in create_batch
    try: return (fa_collate,fa_convert)[self.prebatched](b)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 54, in fa_collate
    else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 54, in <listcomp>
    else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
                     ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/load.py", line 53, in fa_collate
    return (default_collate(t) if isinstance(b, _collate_types)
            ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 398, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 159, in collate
    return collate_fn_map[collate_type](
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 272, in collate_tensor_fn
    return torch.stack(batch, 0, out=out)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py", line 384, in __torch_function__
    res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/_tensor.py", line 1648, in __torch_function__
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Error when trying to collate the data into batches with fa_collate, at least two tensors in the batch are not the same size.

Mismatch found on axis 0 of the batch and is of type `TensorImage`:
	Item at index 0 has shape: torch.Size([3, 1024, 768])
	Item at index 2 has shape: torch.Size([3, 768, 1024])

Please include a transform in `after_item` that ensures all data of type TensorImage is the same size


## === cell 13
_ = learn.validate()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/485790432.py in <cell line: 0>()
----> 1 _ = learn.validate()
      2 

NameError: name 'learn' is not defined

## === cell 14
learn.export(f"{model_name}.pkl", pickle_module=dill)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/172335340.py in <cell line: 0>()
----> 1 learn.export(f"{model_name}.pkl", pickle_module=dill)
      2 

NameError: name 'learn' is not defined

## === cell 15
pass



## === cell 16
pass



## === cell 17
assert "learn" in globals() and learn is not None



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/3281683603.py in <cell line: 0>()
----> 1 assert "learn" in globals() and learn is not None
      2 

AssertionError: 

## === cell 18
pass



## === cell 19
sample_submission = pd.read_csv(sample_sub_path)
submission = sample_submission.copy()

submission["image"] = submission["image"].astype(str)
submission["image_path"] = test_images_dir.as_posix() + "/" + submission["image"]
submission.head()



## === cell 20
test_items = submission["image_path"].tolist()

test_dl = learn.dls.test_dl(
    test_items,
    with_labels=False,
    bs=batch_size,
    num_workers=min(8, os.cpu_count() or 2),
    pin_memory=torch.cuda.is_available(),
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2849809300.py in <cell line: 0>()
      3 test_items = submission["image_path"].tolist()
      4 
----> 5 test_dl = learn.dls.test_dl(
      6     test_items,
      7     with_labels=False,

NameError: name 'learn' is not defined

## === cell 21
pass



## === cell 22
with learn.no_bar(), learn.no_logging():
    probs, _ = learn.get_preds(dl=test_dl)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/984636290.py in <cell line: 0>()
----> 1 with learn.no_bar(), learn.no_logging():
      2     probs, _ = learn.get_preds(dl=test_dl)
      3 

NameError: name 'learn' is not defined

## === cell 23
preds_idx = probs.topk(5, dim=1)[1].cpu().numpy()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2013419085.py in <cell line: 0>()
----> 1 preds_idx = probs.topk(5, dim=1)[1].cpu().numpy()
      2 

NameError: name 'probs' is not defined

## === cell 24
vocab = learn.dls.vocab
preds = [" ".join(str(vocab[i]) for i in row) for row in preds_idx]
preds[:5]



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/382789564.py in <cell line: 0>()
----> 1 vocab = learn.dls.vocab
      2 preds = [" ".join(str(vocab[i]) for i in row) for row in preds_idx]
      3 preds[:5]
      4 

NameError: name 'learn' is not defined

## === cell 25
final_sub = sample_submission.copy()
final_sub["hotel_id"] = preds

out_path = "./kaggle/working/submission.csv"
final_sub.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape:", final_sub.shape)
final_sub.head()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3879794813.py in <cell line: 0>()
      1 final_sub = sample_submission.copy()
----> 2 final_sub["hotel_id"] = preds
      3 
      4 out_path = "./kaggle/working/submission.csv"
      5 final_sub.to_csv(out_path, index=False)

NameError: name 'preds' is not defined
