# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd

import fastai
from fastai.vision.all import *
import dill

import torchvision
import torch

pd.set_option("display.max_columns", 200)

torch.backends.cudnn.benchmark = True
torch.set_num_threads(max(1, os.cpu_count() or 1))

seed = 0
set_seed(seed, reproducible=True)
torch.backends.cudnn.deterministic = True



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
requested_models = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",  # v7
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",  # v8
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",  # v11
]


def existing_paths(paths):
    out = []
    for p in paths:
        pp = Path(p)
        if pp.exists():
            out.append(pp)
    return out


models = existing_paths(requested_models)

if len(models) == 0:
    roots = [Path("../input")]
    found = []
    for r in roots:
        if r.exists():
            found.extend(list(r.rglob("*.pkl")))
    found = sorted(
        found,
        key=lambda p: (
            ("export" not in p.name.lower()) and ("learn" not in p.name.lower()),
            -p.stat().st_size,
        ),
    )
    models = found[:3]  # keep ensemble size similar to original (3 models)

models



## === cell 3
DATA_ROOT = Path("../input/hotel-id-2021-fgvc8")
train_csv = DATA_ROOT / "train.csv"
sample_sub_csv = DATA_ROOT / "sample_submission.csv"
test_img_dir = DATA_ROOT / "test_images"
train_img_dir = DATA_ROOT / "train_images"

submission = pd.read_csv(sample_sub_csv)
test = submission.copy()

test["image_path"] = (test_img_dir.as_posix() + "/" + test["image"].astype(str)).astype(
    str
)

test.head()




## === cell 4
def make_preds_from_probs(probs, vocab):
    preds_idx = probs.topk(5)[1].cpu().numpy()
    return [" ".join(map(str, [vocab[i] for i in row])) for row in preds_idx]


use_models = len(models) > 0
use_models, [str(m) for m in models]




## === cell 5
def _build_image_path_train(row):
    return str(train_img_dir / str(row["chain"]) / str(row["image"]))


@torch.no_grad()
def extract_embeddings_from_dataloader(dl, model, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    embs = []
    with torch.inference_mode():
        for b in dl:
            xb = b[0]
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True)
            else:
                xb = xb.to(device)
            e = model(xb)
            if e.ndim > 2:
                e = torch.flatten(e, 1)
            e = torch.nn.functional.normalize(e, p=2, dim=1)
            embs.append(e.detach().cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def knn_predict_top5(
    train_embs, train_labels, test_embs, k=50, topn=5, chunk_size=2048
):
    """
    Speed-only changes (correctness-preserving):
    - Keep exact same similarity + topk selection.
    - Speed up final voting by using numpy bincount over integer-coded labels
      (equivalent to counting occurrences per hotel_id for each test row).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embs = train_embs.to(device)
    test_embs = test_embs.to(device)

    n_test = test_embs.shape[0]
    n_train = train_embs.shape[0]
    k_eff = min(k, n_train)

    best_vals = torch.full((n_test, k_eff), -1e9, device=device)
    best_idx = torch.full((n_test, k_eff), -1, device=device, dtype=torch.long)

    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk = train_embs[start:end]  # [c, d]
        sims = test_embs @ chunk.T  # [n_test, c]

        cand_vals = torch.cat([best_vals, sims], dim=1)
        cand_idx = torch.cat(
            [
                best_idx,
                torch.arange(start, end, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(n_test, -1),
            ],
            dim=1,
        )

        best_vals, sel = torch.topk(
            cand_vals, k=k_eff, dim=1, largest=True, sorted=True
        )
        best_idx = cand_idx.gather(1, sel)

        del sims, cand_vals, cand_idx, sel

    idx_cpu = best_idx.cpu().numpy()

    train_labels = np.asarray(train_labels, dtype=object)
    uniq, inv = np.unique(train_labels, return_inverse=True)
    inv = inv.astype(np.int32, copy=False)

    out = []
    for row_idx in idx_cpu:
        row_lab = inv[row_idx.astype(np.int64, copy=False)]
        counts = np.bincount(row_lab, minlength=len(uniq))
        nz = np.flatnonzero(counts)
        if nz.size == 0:
            out.append("")
            continue
        items = [(int(counts[i]), str(uniq[i]), uniq[i]) for i in nz]
        items.sort(key=lambda x: (-x[0], x[1]))
        out.append(" ".join(str(t[2]) for t in items[:topn]))
    return out


_TOP5_PRIOR_CACHE = {}


def _top5_global_prior(train_csv_path):
    p = str(train_csv_path)
    if p in _TOP5_PRIOR_CACHE:
        return _TOP5_PRIOR_CACHE[p]
    tr = pd.read_csv(train_csv_path, usecols=["hotel_id"])
    top5 = tr["hotel_id"].astype(str).value_counts().head(5).index.tolist()
    while len(top5) < 5:
        top5.append(top5[-1] if top5 else "0")
    _TOP5_PRIOR_CACHE[p] = top5[:5]
    return _TOP5_PRIOR_CACHE[p]


def _ensure_5_ids(pred_list, pad_ids):
    out = []
    for s in pred_list:
        ids = [x for x in str(s).split() if x != ""]
        seen = set()
        ids = [x for x in ids if not (x in seen or seen.add(x))]
        for p in pad_ids:
            if len(ids) >= 5:
                break
            if p not in ids:
                ids.append(p)
        out.append(" ".join(ids[:5]))
    return out


def fallback_knn_submission(test_df, train_csv_path, max_train_images=60000, seed=0):
    tr = pd.read_csv(train_csv_path, usecols=["image", "chain", "hotel_id"])

    tr["image_path"] = (
        train_img_dir.as_posix()
        + "/"
        + tr["chain"].astype(str)
        + "/"
        + tr["image"].astype(str)
    )

    exists_mask = tr["image_path"].map(os.path.exists)
    tr = tr.loc[exists_mask].reset_index(drop=True)

    if len(tr) == 0:
        prior = _top5_global_prior(train_csv_path)
        return [" ".join(prior)] * len(test_df)

    if len(tr) > max_train_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tr), size=max_train_images, replace=False)
        tr = tr.iloc[np.sort(idx)].reset_index(drop=True)

    arch = torchvision.models.resnet50
    resnet = arch(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = create_body(resnet, pretrained=False, cut=-2)  # outputs a feature map
    backbone.eval()

    item_tfms = [ColReader("image_path"), PILImage.create, Resize(224)]
    batch_tfms = [IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]

    cpu_cnt = os.cpu_count() or 2
    n_workers = min(8, max(2, cpu_cnt // 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dblock = DataBlock(
        blocks=(ImageBlock,),
        get_x=ColReader("image_path"),
        item_tfms=[PILImage.create, Resize(224)],
        batch_tfms=batch_tfms,
    )

    train_dl = dblock.dataloaders(
        tr,
        bs=64,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
    ).test_dl(tr, with_labels=False)
    test_dl = dblock.dataloaders(
        test_df,
        bs=64,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
    ).test_dl(test_df, with_labels=False)

    train_labels = tr["hotel_id"].astype(str).tolist()

    train_embs = extract_embeddings_from_dataloader(train_dl, backbone, device=device)
    test_embs = extract_embeddings_from_dataloader(test_dl, backbone, device=device)

    preds = knn_predict_top5(
        train_embs, train_labels, test_embs, k=50, topn=5, chunk_size=2048
    )
    preds = _ensure_5_ids(preds, _top5_global_prior(train_csv_path))
    return preds


@torch.no_grad()
def manual_tta_probs(learn, dl, n=6):
    learn.model.eval()
    device = learn.dls.device
    n_classes = len(learn.dls.vocab)
    ds_len = len(dl.dataset)

    probs_sum = torch.zeros((ds_len, n_classes), dtype=torch.float32)
    use_cuda = device.type == "cuda"
    after_batch = dl.after_batch
    model = learn.model

    with torch.inference_mode():
        for _ in range(n):
            offset = 0
            for b in dl:
                xb = b[0]
                bs = xb.shape[0]
                if use_cuda:
                    xb = xb.to(device, non_blocking=True)
                else:
                    xb = xb.to(device)
                xb_aug = after_batch(xb)
                pred = model(xb_aug)
                probs_sum[offset : offset + bs].add_(
                    torch.softmax(pred, dim=1).detach().cpu()
                )
                offset += bs

    probs_sum.div_(float(n))
    return probs_sum




## === cell 6
preds = None

if use_models:
    probs = None
    last_learn = None

    cpu_cnt = os.cpu_count() or 2
    n_workers = min(8, max(2, cpu_cnt // 2))

    test_df = test

    for model_path in models:
        cpu_flag = not torch.cuda.is_available()
        learn = load_learner(fname=Path(model_path), cpu=cpu_flag, pickle_module=dill)
        last_learn = learn
        learn.model.eval()

        test_dl = learn.dls.test_dl(
            test_df,
            with_labels=False,
            item_tfms=[ColReader("image_path"), PILImage.create],
            num_workers=n_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(n_workers > 0),
            prefetch_factor=4 if n_workers > 0 else None,
        )

        probs_temp = manual_tta_probs(learn, test_dl, n=6)

        if probs is None:
            probs = probs_temp
        else:
            probs += probs_temp

        del learn, test_dl, probs_temp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probs = probs / len(models)
    preds = make_preds_from_probs(probs, last_learn.dls.vocab)
else:
    preds = fallback_knn_submission(test, train_csv, max_train_images=60000, seed=0)

preds[:3], len(preds)



## === cell 7
out = submission.copy()
out["hotel_id"] = preds

out["image"] = out["image"].astype(str)
out["hotel_id"] = out["hotel_id"].astype(str)

pad5 = _top5_global_prior(train_csv)
out["hotel_id"] = _ensure_5_ids(out["hotel_id"].tolist(), pad5)

out_path = Path("submission.csv")
out.to_csv(out_path, index=False)

out.head(), str(out_path), out.shape
