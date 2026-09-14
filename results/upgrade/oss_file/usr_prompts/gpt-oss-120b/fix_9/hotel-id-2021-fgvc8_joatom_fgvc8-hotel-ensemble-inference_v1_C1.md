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

0.5955733771154317

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The changes increase batch size for train‑embedding extraction, pre‑allocate tensors instead of appending Python lists, replace `torch.no_grad()` with the faster `torch.inference_mode()`, and keep everything else identical so the predictions remain exactly the same while reducing Python‑level overhead and memory copies, allowing the whole script to finish well under the 600‑second limit.'

# 9. Code solution

## === cell 0
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pathlib import Path
import pandas as pd
from fastai.vision.all import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_float32_matmul_precision("high")  # ensure fast matmul on supported hardware
torch.backends.cuda.matmul.allow_tf32 = (
    True  # enable TF32 for faster matmul on Ampere GPUs
)



## === cell 1
base_path = Path("../input/hotel-id-2021-fgvc8")
train_csv = base_path / "train.csv"
test_csv = base_path / "sample_submission.csv"
train_img_path = base_path / "train_images"
test_img_path = base_path / "test_images"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)




## === cell 2
def get_train_img_path(row):
    return train_img_path / str(row["chain"]) / row["image"]


import os

num_workers = min(8, os.cpu_count() or 1)

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=get_train_img_path,
    get_y=ColReader("hotel_id"),
    splitter=IndexSplitter([]),  # all data used for embedding extraction
    item_tfms=Resize(224, method="crop"),
)

dls = dblock.dataloaders(
    train_df,
    bs=512,  # larger batch reduces number of forward passes
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    batch_tfms=Normalize.from_stats(*imagenet_stats),
)



## === cell 3
learn = cnn_learner(dls, resnet34, pretrained=True, loss_func=CrossEntropyLoss())
learn.model = learn.model[0]  # strip the head, keep the body only
learn.model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learn.model.to(device)



## === cell 4
emb_path = Path("train_embeddings.pt")
lbl_path = Path("train_labels.pt")

if emb_path.is_file() and lbl_path.is_file():
    train_embeddings = torch.load(emb_path, map_location=device)
    train_labels = torch.load(lbl_path, map_location="cpu")
else:
    N = len(train_df)
    train_embeddings = torch.empty((N, 512), dtype=torch.float16, device=device)
    train_labels = torch.empty(N, dtype=torch.int64, device="cpu")
    idx = 0
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for xb, yb in dls.train:
            xb = xb.to(device)
            feats = learn.model(xb)  # (bs, 512)
            feats = F.normalize(feats, p=2, dim=1)
            bs = feats.shape[0]
            train_embeddings[idx : idx + bs] = feats.half()
            train_labels[idx : idx + bs] = yb.cpu()
            idx += bs
    torch.save(train_embeddings, emb_path)
    torch.save(train_labels, lbl_path)

train_embeddings_T = train_embeddings.t().contiguous()  # (512, N_train) on device

vocab = dls.vocab  # list of hotel_id strings ordered by class index



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1977477784.py in <cell line: 0>()
     18             feats = F.normalize(feats, p=2, dim=1)
     19             bs = feats.shape[0]
---> 20             train_embeddings[idx : idx + bs] = feats.half()
     21             train_labels[idx : idx + bs] = yb.cpu()
     22             idx += bs

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

RuntimeError: expand(torch.cuda.HalfTensor{[512, 512, 7, 7]}, size=[512, 512]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (4)

## === cell 5
test_image_paths = [test_img_path / fname for fname in test_df["image"].values]

test_dl = dls.test_dl(
    test_image_paths,
    bs=1024,  # larger batch reduces loop overhead
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

preds_list = [None] * len(test_df)
idx = 0

with torch.inference_mode(), torch.cuda.amp.autocast():
    for xb in test_dl:  # xb already transformed
        xb = xb.to(device)
        feats = learn.model(xb)  # (bs, 512)
        feats = F.normalize(feats, p=2, dim=1)
        feats = feats.half()
        sims = torch.mm(feats, train_embeddings_T)  # (bs, N_train)
        _, topk_idx = sims.topk(5, dim=1)  # (bs, 5)
        topk_cpu = topk_idx.cpu().numpy()
        batch_preds = [" ".join(vocab[i] for i in row) for row in topk_cpu]
        batch_len = len(batch_preds)
        preds_list[idx : idx + batch_len] = batch_preds
        idx += batch_len



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1684937249.py in <cell line: 0>()
     14 with torch.inference_mode(), torch.cuda.amp.autocast():
     15     for xb in test_dl:  # xb already transformed
---> 16         xb = xb.to(device)
     17         feats = learn.model(xb)  # (bs, 512)
     18         feats = F.normalize(feats, p=2, dim=1)

AttributeError: 'tuple' object has no attribute 'to'

## === cell 6
submission = pd.read_csv(test_csv)  # columns: image, hotel_id (placeholder)
submission["hotel_id"] = preds_list
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(submission.head())
