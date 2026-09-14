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

os.makedirs("./kaggle/working", exist_ok=True)



## === cell 1
import warnings
import random

import torch
import numpy as np
import pandas as pd

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
sample_paths = train_data["image_path"].sample(25, random_state=0).tolist()
missing = sum(0 if os.path.exists(p) else 1 for p in sample_paths)
assert (
    missing == 0
), f"Found {missing} missing image paths in sample; check directory structure."



## === cell 7
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
set_seed(42, reproducible=True)

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



## === cell 8
if torch.cuda.is_available():
    batch_size = 64
else:
    batch_size = 32


def _suggest_num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(0, min(4, cpu))


NUM_WORKERS = _suggest_num_workers()


def _dl_kwargs(num_workers: int):
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def create_dataLoader():
    df = train_data[["image_path", "hotel_id"]]

    item_tfms = [Resize(224, method="pad", pad_mode="reflection")]
    batch_tfms = list(aug_transforms(size=224))

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
        **_dl_kwargs(NUM_WORKERS),
    )
    return dls




## === cell 9
dataset = create_dataLoader()
dataset



## === cell 10
pass




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
model_pth = Path(f"{model_name}.pth")
model_pkl = Path(f"{model_name}.pkl")

learn = None

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    if model_pkl.exists():
        learn = load_learner(
            model_pkl, cpu=not torch.cuda.is_available(), pickle_module=dill
        )
    else:
        learn = cnn_learner(
            dataset,
            models.resnet101,
            metrics=[accuracy, top_k_accuracy],
            opt_func=QHAdam,
        ).to_fp16()

        if model_pth.exists():
            learn.load(model_name)
        else:
            raise RuntimeError(
                "No pre-trained model found (missing .pkl and .pth). "
                "Training from scratch/fine-tuning here will exceed the 600-second limit. "
                "Please provide model_allData_rn101.pth or model_allData_rn101.pkl in the working directory."
            )



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3438143737.py in <cell line: 0>()
     26             learn.load(model_name)
     27         else:
---> 28             raise RuntimeError(
     29                 "No pre-trained model found (missing .pkl and .pth). "
     30                 "Training from scratch/fine-tuning here will exceed the 600-second limit. "

RuntimeError: No pre-trained model found (missing .pkl and .pth). Training from scratch/fine-tuning here will exceed the 600-second limit. Please provide model_allData_rn101.pth or model_allData_rn101.pkl in the working directory.

## === cell 13
_ = learn.validate()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/485790432.py in <cell line: 0>()
----> 1 _ = learn.validate()
      2 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in validate(self, ds_idx, dl, cbs)
    284     def validate(self, ds_idx=1, dl=None, cbs=None):
    285         if dl is None: dl = self.dls[ds_idx]
--> 286         with self.validation_context(cbs=cbs): self._do_epoch_validate(ds_idx, dl)
    287         return getattr(self, 'final_record', None)
    288 

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
   1453                 data = self._task_info.pop(self._rcvd_idx)[1]
   1454                 self._rcvd_idx += 1
-> 1455                 return self._process_data(data)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0

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

KeyError: Caught KeyError in DataLoader worker process 2.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py", line 263, in encodes
    return TensorCategory(self.vocab.o2i[o])
                          ~~~~~~~~~~~~~~^^^
KeyError: 53582

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
KeyError: "Label '53582' was not included in the training dataset"


## === cell 14
if not model_pkl.exists():
    learn.export(model_pkl.as_posix(), pickle_module=dill)



## === cell 15
pass



## === cell 16
pass



## === cell 17
assert "learn" in globals() and learn is not None



## === cell 18
pass



## === cell 19
sample_submission = pd.read_csv(sample_sub_path)
submission = sample_submission  # avoid copy; build needed columns only
submission["image"] = submission["image"].astype(str)
submission["image_path"] = test_images_dir.as_posix() + "/" + submission["image"]
submission.head()



## === cell 20
test_items = submission["image_path"].to_list()

test_dl = learn.dls.test_dl(
    test_items,
    with_labels=False,
    bs=batch_size,
    **_dl_kwargs(NUM_WORKERS),
)



## === cell 21
pass



## === cell 22
with learn.no_bar(), learn.no_logging():
    probs, _ = learn.get_preds(dl=test_dl)



## === cell 23
preds_idx = probs.topk(5, dim=1)[1].cpu().numpy()



## === cell 24
vocab = np.asarray(learn.dls.vocab, dtype=object)
preds = [" ".join(map(str, vocab[row])) for row in preds_idx]
preds[:5]



## === cell 25
final_sub = sample_submission.copy()
final_sub["hotel_id"] = preds

out_path = "./kaggle/working/submission.csv"
final_sub.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape:", final_sub.shape)
final_sub.head()
