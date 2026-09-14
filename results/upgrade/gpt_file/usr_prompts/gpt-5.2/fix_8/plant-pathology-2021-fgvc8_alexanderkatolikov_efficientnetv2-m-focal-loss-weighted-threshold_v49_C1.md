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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7868144044321331

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'Your notebook currently can’t yield a valid submission because it relies on extra Kaggle input datasets (`../input/train-labeled/...`, `../input/resnet-model/...`, and `../input/torchcontrib/...`) that are not present in the provided file paths, so it error before writing `submission.csv`. I make the smallest changes needed to (1) remove those missing dependencies while preserving the same core model architecture (ResNeXt101 with a replaced FC), (2) fix the invalid `torch.exp(logits)` usage for multi-class (use `softmax` instead), and (3) ensure the submission rows are aligned to `sample_submission.csv` order and have space-delimited label strings. This produce a valid `submission.csv` end-to-end and should improve score versus an invalid/no-submission state, without changing the overall approach (single-label classification with argmax).'
- What this solution (achieved 0.12239) has done: 'The main timeout risk here is extreme per-image overhead (PIL decode + transforms) combined with an enormous ResNeXt-101 at 640px and a test loop running single-image batches; we keep the exact model and transforms but remove needless overhead by (1) fixing the test directory listing bug, (2) using faster/safer image loading + dataloader settings (workers, pin_memory, persistent_workers, prefetch), and (3) batching test inference (same predictions, far fewer forward passes). Additionally, we make the metric helper functions O(1) via precomputed label-token sets (identical semantics) and avoid repeated pandas lookups. No training loop is introduced/changed; core logic and outputs stay the same aside from negligible floating-point differences.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by extremely expensive per-epoch training of a huge ResNeXt-101 at 640×640, plus repeated CPU image decode/resize/crop and disk I/O from per-sample tensor caching. To finish under 600 seconds without changing the model, training loop, transforms, or loss, I’m keeping the exact algorithm but removing the slow disk cache writes/reads (they add heavy overhead and don’t help within a single run), switching DataLoader workers to a setting that avoids CPU oversubscription, and enabling faster host→GPU transfer with pinned memory + non-blocking moves for the CPU path. I also add a one-time warmup for `torch.compile` to reduce compile/graph-capture overhead during the first real step, and use cuDNN TF32 (matmul/conv) which preserves semantics with only negligible floating-point differences while dramatically speeding up Ampere+ GPUs. Paths, splits, model architecture, epochs, and thresholds remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True  # safe for fixed-size inputs; speeds convs
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"

_CPU = os.cpu_count() or 2
_NUM_WORKERS = 4 if _CPU >= 8 else 2
_PIN_MEMORY = torch.cuda.is_available()

_CACHE_TRAIN = False
_CACHE_VAL = False
_CACHE_TEST = False

CACHE_ROOT = Path("./__cache_pp2021__")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)



## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train_df



## === cell 3
train_df["labels"].value_counts()



## === cell 4
all_tokens = sorted(
    {t for s in train_df["labels"].astype(str).values for t in s.split(" ") if t}
)
token2id = {t: i for i, t in enumerate(all_tokens)}
id2token = {i: t for t, i in token2id.items()}
NUM_CL = len(all_tokens)
NUM_CL




## === cell 5
def labels_to_multihot(label_str, num_classes=NUM_CL):
    y = np.zeros(num_classes, dtype=np.float32)
    for t in str(label_str).split(" "):
        if t:
            y[token2id[t]] = 1.0
    return y


train_df["y"] = train_df["labels"].apply(labels_to_multihot)
train_df[["image", "labels"]].head()



## === cell 6
tr_df = train_df[:trainnum].reset_index(drop=True)
print(len(tr_df))
X_Train = tr_df["image"].values
Y_Train = np.stack(tr_df["y"].values)



## === cell 7
Transform = transforms.Compose(
    [
        transforms.Resize(
            (IM_SIZE, IM_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)



## === cell 8
Transformval = transforms.Compose(
    [
        transforms.Resize(
            (IM_SIZE, IM_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 9
def _safe_stem(s: str) -> str:
    return s.replace("/", "_")


def _cache_dir_for(split: str, im_size: int, crop: int = None) -> Path:
    if crop is None:
        name = f"{split}_resize{im_size}"
    else:
        name = f"{split}_resize{im_size}_centercrop{crop}"
    d = CACHE_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


class GetData(Dataset):
    def __init__(
        self, Dir, FNames, Labels, Transform, cache=False, cache_dir: Path = None
    ):
        self.dir = Dir
        self.fnames = list(FNames)
        self.transform = Transform
        self.labels = Labels
        self.cache = bool(cache)
        self.cache_dir = cache_dir if (self.cache and cache_dir is not None) else None

        self._paths = (
            [os.path.join(self.dir, fn) for fn in self.fnames] if self.fnames else []
        )

        dlow = str(self.dir).lower()
        self._is_train = "train" in dlow
        self._is_test = "test" in dlow

    def __len__(self):
        return len(self.fnames)

    def _tensor_cache_path(self, index: int) -> Path:
        return self.cache_dir / f"{_safe_stem(self.fnames[index])}.pt"

    def _load_and_transform(self, index):
        fp = self._paths[index]
        with Image.open(fp) as img:
            x = img.convert("RGB")
        x = self.transform(x)
        return x

    def _get_x(self, index):
        if self.cache_dir is None:
            return self._load_and_transform(index)

        p = self._tensor_cache_path(index)
        if p.exists():
            return torch.load(p, map_location="cpu")
        x = self._load_and_transform(index)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(x, tmp)
        os.replace(tmp, p)
        return x

    def __getitem__(self, index):
        x = self._get_x(index)

        if self._is_train:
            y = torch.from_numpy(self.labels[index])
            return x, y
        elif self._is_test:
            return x, self.fnames[index]




## === cell 10
trainset = GetData(
    TRAIN_DIR,
    X_Train,
    Y_Train,
    Transform,
    cache=_CACHE_TRAIN,
    cache_dir=_cache_dir_for("train", IM_SIZE, crop=None),
)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=(2 if _NUM_WORKERS > 0 else None),
)



## === cell 11
val_df = train_df[-valnum:].reset_index(drop=True)
print(len(val_df))
X_val = val_df["image"].values
Y_val = np.stack(val_df["y"].values)

valset = GetData(
    TRAIN_DIR,
    X_val,
    Y_val,
    Transformval,
    cache=_CACHE_VAL,
    cache_dir=_cache_dir_for("val", IM_SIZE, crop=int(IM_SIZE * 0.8)),
)
valloader = DataLoader(
    valset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)



## === cell 12
next(iter(trainloader))[0].shape




## === cell 13
def PREDS_FROM_PROBS(probs, thr=0.5):
    idx = (probs >= thr).nonzero(as_tuple=False).view(-1).tolist()
    if len(idx) == 0:
        return ["healthy"]
    toks = [id2token[i] for i in idx]
    return toks




## === cell 14
def micro_f1_from_logits(logits, targets, thr=0.5, eps=1e-9):
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).to(targets.dtype)
    tp = (pred * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    fn = ((1 - pred) * targets).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1, precision, recall




## === cell 15
model = torchvision.models.resnext101_32x8d(weights=None)
model.fc = nn.Linear(2048, NUM_CL, bias=True)

model = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    except Exception as _e:
        pass




## === cell 16
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        stream = self.stream

        def _preload():
            nonlocal next_batch
            try:
                batch = next(it)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(stream):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    a, b = batch
                    a = a.to(self.device, non_blocking=True)
                    if torch.is_tensor(b):
                        b = b.to(self.device, non_blocking=True)
                    next_batch = (a, b)
                else:
                    next_batch = batch

        next_batch = None
        _preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_batch
            _preload()
            yield batch




## === cell 17
if DEVICE.type == "cuda":
    model.eval()
    with torch.inference_mode():
        dummy = torch.zeros((1, 3, IM_SIZE, IM_SIZE), device=DEVICE)
        _ = model(dummy)
        del dummy
        torch.cuda.synchronize()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n = 0

    train_iterable = (
        CUDAPrefetcher(trainloader, DEVICE) if DEVICE.type == "cuda" else trainloader
    )

    for imgs, ys in train_iterable:
        if DEVICE.type != "cuda":
            imgs = imgs.to(DEVICE, non_blocking=_PIN_MEMORY)
            ys = ys.to(DEVICE, non_blocking=_PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n += bs

    train_loss = running_loss / max(1, n)

    model.eval()
    with torch.inference_mode():
        vloss = 0.0
        vn = 0

        val_iterable = (
            CUDAPrefetcher(valloader, DEVICE) if DEVICE.type == "cuda" else valloader
        )

        for imgs, ys in val_iterable:
            if DEVICE.type != "cuda":
                imgs = imgs.to(DEVICE, non_blocking=_PIN_MEMORY)
                ys = ys.to(DEVICE, non_blocking=_PIN_MEMORY)
            logits = model(imgs)
            loss = criterion(logits, ys)
            bs = imgs.size(0)
            vloss += loss.item() * bs
            vn += bs
        val_loss = vloss / max(1, vn)

    print(
        f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f}"
    )



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4144699381.py in <cell line: 0>()
     24 
     25         optimizer.zero_grad(set_to_none=True)
---> 26         logits = model(imgs)
     27         loss = criterion(logits, ys)
     28         loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in forward(self, x)
    282         return x
    283 
--> 284     def forward(self, x: Tensor) -> Tensor:
    285         return self._forward_impl(x)
    286 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    743             )
    744             try:
--> 745                 return fn(*args, **kwargs)
    746             finally:
    747                 _maybe_set_eval_frame(prior)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in forward(*runtime_args)
   1182         full_args.extend(params_flat)
   1183         full_args.extend(runtime_args)
-> 1184         return compiled_fn(full_args)
   1185 
   1186     # Just for convenience

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in runtime_wrapper(args)
    308                 True
    309             ), torch.enable_grad():
--> 310                 all_outs = call_func_at_runtime_with_args(
    311                     compiled_fn, args_, disable_amp=disable_amp, steal_args=True
    312                 )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in g(args)
     98 def make_boxed_func(f):
     99     def g(args):
--> 100         return f(*args)
    101 
    102     g._boxed_call = True  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/torch/autograd/function.py in apply(cls, *args, **kwargs)
    573             # See NOTE: [functorch vjp and autograd interaction]
    574             args = _functorch.utils.unwrap_dead_wrappers(args)
--> 575             return super().apply(*args, **kwargs)  # type: ignore[misc]
    576 
    577         if not is_setup_ctx_defined:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in forward(ctx, *deduped_flat_tensor_args)
   1583                 # - Note that donated buffer logic requires (*saved_tensors, *saved_symints) showing up last
   1584                 #   in the fw output order.
-> 1585                 fw_outs = call_func_at_runtime_with_args(
   1586                     CompiledFunction.compiled_fw,
   1587                     args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in wrapper(runtime_args)
    488                 )
    489                 return out
--> 490             return compiled_fn(runtime_args)
    491 
    492         return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in inner_fn(args)
    670                 old_args.clear()
    671 
--> 672             outs = compiled_fn(args)
    673 
    674             # Inductor cache DummyModule can return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/output_code.py in __call__(self, inputs)
    464         assert self.current_callable is not None
    465         try:
--> 466             return self.current_callable(inputs)
    467         finally:
    468             AutotuneCacheBundler.end_compile()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in run(new_inputs)
   1206             ), dynamo_utils.preserve_rng_state():
   1207                 compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
-> 1208         return compiled_fn(new_inputs)
   1209 
   1210     return run

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in deferred_cudagraphify(inputs)
    380         fn = fn_cache.get(int_key)
    381         if fn is not None:
--> 382             return fn(inputs)
    383 
    384         if int_key is None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/utils.py in run(new_inputs)
   2126     def run(new_inputs: List[InputType]):
   2127         copy_misaligned_inputs(new_inputs, inputs_to_check)
-> 2128         return model(new_inputs)
   2129 
   2130     return run

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run(self, new_inputs, function_id)
   1945         assert self.graph is not None, "Running CUDAGraph after shutdown"
   1946         self.mode = self.id_to_mode[function_id]
-> 1947         out = self._run(new_inputs, function_id)
   1948 
   1949         # The forwards are only pending following invocation, not before

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in _run(self, new_inputs, function_id)
   2125             log_pt2_compile_event=True,
   2126         ):
-> 2127             out = self.record_function(new_inputs, function_id)
   2128 
   2129         return out

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in record_function(self, new_inputs, function_id)
   2161         )
   2162         torch.cuda.synchronize()
-> 2163         node = CUDAGraphNode(
   2164             self.ids_to_funcs[function_id],
   2165             graph_id,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in __init__(***failed resolving arguments***)
    981 
    982         # Cleared after recording
--> 983         self.recording_outputs: Optional[OutputType] = self._record(
    984             wrapped_function.model, recording_inputs
    985         )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in _record(self, model, inputs)
   1207             check_memory_pool(self.device, self.cuda_graphs_pool, memory)
   1208 
-> 1209         with preserve_rng_state(), torch.cuda.device(
   1210             self.device
   1211         ), clear_cublas_manager(), torch.cuda.graph(

/usr/local/lib/python3.11/dist-packages/torch/cuda/graphs.py in __enter__(self)
    179         self.stream_ctx.__enter__()
    180 
--> 181         self.cuda_graph.capture_begin(
    182             *self.pool, capture_error_mode=self.capture_error_mode
    183         )

/usr/local/lib/python3.11/dist-packages/torch/cuda/graphs.py in capture_begin(self, pool, capture_error_mode)
     71                 unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_
     72         """  # noqa: B950
---> 73         super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
     74 
     75     def capture_end(self):

RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.You can make a clone to get a normal tensor before doing inplace update.See https://github.com/pytorch/rfcs/pull/17 for more details.

## === cell 18
X_Test = sorted(
    [
        name
        for name in os.listdir(TEST_DIR)
        if os.path.isfile(os.path.join(TEST_DIR, name))
    ]
)



## === cell 19
testset = GetData(
    TEST_DIR,
    X_Test,
    None,
    Transformval,
    cache=_CACHE_TEST,
    cache_dir=_cache_dir_for("test", IM_SIZE, crop=int(IM_SIZE * 0.8)),
)
testloader = DataLoader(
    testset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)



## === cell 20
test_iterable = (
    CUDAPrefetcher(testloader, DEVICE) if DEVICE.type == "cuda" else testloader
)

s_ls = []
THR = 0.5

with torch.inference_mode():
    model.eval()
    for image, fname in test_iterable:
        if DEVICE.type != "cuda":
            image = image.to(DEVICE, non_blocking=_PIN_MEMORY)
        logits = model(image)
        probs = torch.sigmoid(logits).detach().cpu()
        for fn, pr in zip(fname, probs):
            toks = PREDS_FROM_PROBS(pr, thr=THR)
            s_ls.append([fn, " ".join(toks)])



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_55/3532675763.py in <cell line: 0>()
     11         if DEVICE.type != "cuda":
     12             image = image.to(DEVICE, non_blocking=_PIN_MEMORY)
---> 13         logits = model(image)
     14         probs = torch.sigmoid(logits).detach().cpu()
     15         for fn, pr in zip(fname, probs):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1099             # optimization to generate better code in a common case
   1100             self.add_output_instructions(
-> 1101                 self.compile_and_call_fx_graph(
   1102                     tx, list(reversed(stack_values)), root, output_replacements
   1103                 )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1550     if config_patches:
   1551         with config.patch(config_patches):
-> 1552             return compile_fx(
   1553                 model_,
   1554                 example_inputs_,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_base(flat_fn, flat_args, aot_config, fw_metadata)
    201                 assert isinstance(fw_module, GraphModule)
    202                 tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
--> 203             compiled_fw = compiler(fw_module, updated_flat_args)
    204 
    205         if fakified_out_wrapper.needs_post_compile:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in __call__(self, gm, example_inputs)
    487         example_inputs: Sequence[InputType],
    488     ) -> OutputCode:
--> 489         return self.compiler_fn(gm, example_inputs)
    490 
    491 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fw_compiler_base(gm, example_inputs, is_inference)
   1739                     model_outputs_node.meta["user_visible_output_idxs"] = []
   1740 
-> 1741                 return inner_compile(
   1742                     gm,
   1743                     example_inputs,

/usr/lib/python3.11/contextlib.py in inner(*args, **kwds)
     79         def inner(*args, **kwds):
     80             with self._recreate_cm():
---> 81                 return func(*args, **kwds)
     82         return inner
     83 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx_inner(gm, example_inputs, **kwargs)
    567         )
    568 
--> 569         return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
    570             gm,
    571             example_inputs,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_aot.py in debug_wrapper(gm, example_inputs, **kwargs)
    100             # Call the compiler_fn - which is either aot_autograd or inductor
    101             # with fake inputs
--> 102             inner_compiled_fn = compiler_fn(gm, example_inputs)
    103         except Exception as e:
    104             # TODO: Failures here are troublesome because no real inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _compile_fx_inner(gm, example_inputs, **graph_kwargs)
    683             TritonBundler.begin_compile()
    684             try:
--> 685                 mb_compiled_graph = fx_codegen_and_compile(
    686                     gm, example_inputs, inputs_to_check, **graph_kwargs
    687                 )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fx_codegen_and_compile(gm, example_inputs, inputs_to_check, **graph_kwargs)
   1127     scheme: FxCompile = _InProcessFxCompile()
   1128 
-> 1129     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
   1130 
   1131 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in codegen_and_compile(self, gm, example_inputs, inputs_to_check, graph_kwargs)
   1042                                 )
   1043                         else:
-> 1044                             compiled_fn = graph.compile_to_module().call
   1045 
   1046                     num_bytes, nodes_num_elem, node_runtimes = graph.count_bytes()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in compile_to_module(self)
   2025             dynamo_compile_column_us="inductor_code_gen_cumulative_compile_time_us",
   2026         ):
-> 2027             return self._compile_to_module()
   2028 
   2029     def _compile_to_module(self) -> ModuleType:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in _compile_to_module(self)
   2031 
   2032         code, linemap = (
-> 2033             self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
   2034         )
   2035         if config.triton.autotune_at_compile_time:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in codegen(self)
   1962             self.init_wrapper_code()
   1963 
-> 1964             self.scheduler = Scheduler(self.operations)
   1965             V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)
   1966 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in __init__(self, nodes)
   1796     def __init__(self, nodes: List[ir.Operation]) -> None:
   1797         with dynamo_timed("Scheduler.__init__"):
-> 1798             self._init(nodes)
   1799 
   1800     def _init(self, nodes: List[ir.Operation]) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in _init(self, nodes)
   1880             )
   1881         self.merge_loops()
-> 1882         self.finalize_multi_template_buffers()
   1883         if config.reorder_for_compute_comm_overlap:
   1884             self.nodes = comms.reorder_compute_and_comm_for_overlap(self.nodes)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in finalize_multi_template_buffers(self)
   2449                 multi_node = node.node
   2450                 if not config.test_configs.force_extern_kernel_in_multi_template:
-> 2451                     min_node_unfused, _ = multi_node.get_min_choice()
   2452                 else:
   2453                     min_node_unfused = next(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/ir.py in get_min_choice(self)
   4369 
   4370     def get_min_choice(self) -> Tuple[ChoiceCaller, float]:
-> 4371         min_choice = min(self.choice_timings, key=self.choice_timings.get)  # type: ignore[arg-type]
   4372         return (min_choice, self.choice_timings[min_choice])
   4373 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/ir.py in choice_timings(self)
   4347     def choice_timings(self) -> Dict[ChoiceCaller, float]:
   4348         if self._choice_timings is None:
-> 4349             self._choice_timings = self._choice_timings_fn()
   4350         return self._choice_timings
   4351 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in get_timings()
   1543 
   1544             def get_timings():
-> 1545                 timings = do_autotuning(precompile_fn)
   1546                 min_extern_choice = float("inf")
   1547                 for choice, timing in timings.items():

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in do_autotuning(precompile_fn)
   1508 
   1509             autotune_start_ts = time.time()
-> 1510             timings = self.lookup(
   1511                 choices,
   1512                 name,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codecache.py in lookup(self, choices, op, inputs, benchmark)
    369                     # catch and log autotuning failures
    370                     log_errors(e)
--> 371                     raise e
    372 
    373                 self.update_local_cache(local_cache)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codecache.py in lookup(self, choices, op, inputs, benchmark)
    360                 try:
    361                     # re-benchmark everything to try to get consistent numbers from the same machine
--> 362                     timings = benchmark(choices)
    363                     assert all(choice in timings for choice in choices)
    364                     local_cache.setdefault(op, {})

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in autotune(choices)
   1493         def autotune(choices):
   1494             with dynamo_timed(f"{name}_template_autotuning"):
-> 1495                 return make_benchmark_fn()(choices)
   1496 
   1497         if config.autotune_in_subproc:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in benchmark_in_current_process(choices)
   1659             choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]],
   1660         ) -> Dict[Union[ExternKernelCaller, TritonTemplateCaller], float]:
-> 1661             inputs = get_inputs(choices)
   1662             timings = {}
   1663             for choice in choices:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in get_inputs(choices)
   1595         ) -> AutotuneArgs:
   1596             # de-duplicate args
-> 1597             unique_example_inputs = {
   1598                 x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x)
   1599                 for i, x in enumerate(input_nodes)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in <dictcomp>(.0)
   1596             # de-duplicate args
   1597             unique_example_inputs = {
-> 1598                 x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x)
   1599                 for i, x in enumerate(input_nodes)
   1600             }

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in benchmark_example_value(node)
   1834         if isinstance(node, ir.BaseView):
   1835             node = node.unwrap_view()
-> 1836         return AlgorithmSelectorCache.generate_example_value(
   1837             V.graph.sizevars.size_hints(
   1838                 node.get_size(),

/usr/local/lib/python3.11/dist-packages/torch/_inductor/select_algorithm.py in generate_example_value(size, stride, device, dtype, extra_size)
   1853         # the rng states for the real model code.
   1854         with preserve_rng_state():
-> 1855             return rand_strided(
   1856                 size,
   1857                 stride,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/testing.py in rand_strided(size, stride, dtype, device, extra_size)
    390             )
    391         else:
--> 392             buffer = torch.randn(needed_size, dtype=dtype, device=device)
    393     else:
    394         buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 21
pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "labels"])
pred_df



## === cell 22
pred_df.head()



## === cell 23
sub = pred_df[["image", "labels"]].copy()

sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
sub = sample_sub[["image"]].merge(sub, on="image", how="left")

sub["labels"] = sub["labels"].fillna("healthy").astype(str)
sub.loc[sub["labels"].str.strip().eq(""), "labels"] = "healthy"

sub.head()



## === cell 24
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
