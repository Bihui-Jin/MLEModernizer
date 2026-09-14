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

0.6019996632146147

# 6. Current score

0.00117

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'The crash is caused by missing exported model `.pkl` files (`../input/fgvc8hotel/...`) in this Kaggle environment, so the inference loop never runs and `probs` stays `None`. I fix the pipeline to (1) automatically discover any available `.pkl` learners under `../input/` and use them if present, and (2) otherwise fall back to a deterministic “prior” baseline that outputs the 5 most frequent `hotel_id` values from `train.csv` (producing a valid submission and a non-trivial MAP@5). I also fix the test dataframe so fastai’s `test_dl` correctly reads images via a `PILImage.create` transform from the full file paths. Finally, I ensure the submission is written as `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should increase MAP@5 with the smallest safe changes that preserve your inference/ensemble logic. The main issue is that your notebook almost certainly falls back to the “top-5 prior” because no `.pkl` exports exist in this dataset; that baseline yields the very low score you see. I keep your ensemble code intact, but add a deterministic, stronger fallback that still matches the competition semantics: build image embeddings with a pretrained torchvision backbone (no training loop) and do kNN retrieval against a limited set of train images to produce the top-5 hotel_ids per test image. This is a minimal addition (only used when no `.pkl` is found), runs within the time limit by subsampling the index set, and writes the same `submission.csv` schema.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime error by importing `torchvision`, which is required by the kNN fallback path. I also make the fallback embedding extractor explicitly use the same device for both train/test embeddings to avoid accidental CPU/GPU mismatches, while keeping your retrieval logic unchanged. Finally, I add a small safety guard to ensure we always output exactly 5 hotel IDs per test image (padding with global priors if kNN returns fewer), so the submission is always valid and score-stable.'
- What this solution (achieved 0.00117) has done: 'I fix the crash in the kNN fallback by instantiating the torchvision ResNet50 model before passing it to `fastai.create_body` (it currently passes the function itself, causing the `.children()` AttributeError). I also add a small safety guard so if model discovery finds zero usable `.pkl` learners, the pipeline reliably falls back to kNN and still produces predictions for every test row. Finally, I keep your retrieval/embedding logic unchanged, only ensuring the submission is always written with the correct columns and exactly 5 space-delimited hotel IDs per image.'
- What this solution (achieved 0.00117) has done: 'I fix the device-mismatch crash in the kNN fallback by ensuring the normalization stats tensors live on the same device as the image tensors (and by moving the backbone to the chosen device). This keeps your retrieval logic identical while making the embedding extraction run end-to-end on GPU/CPU reliably. Since your current score is far below target and the model `.pkl` path often isn’t available, getting the kNN fallback working is the minimal change that should substantially increase MAP@5 versus the global-prior baseline. I also keep the submission-writing logic intact and ensure we still always output exactly 5 IDs per test image.'
- What this solution (achieved 0.00117) has done: 'I fix the kNN fallback crash by ensuring the `Normalize` transform’s `mean/std` are moved onto the same device as the image batch **before** the pipeline applies normalization (your current fix runs too late, after `tfm(im)` already fails). I keep your retrieval/model logic intact and only adjust `extract_embeddings` to apply transforms step-by-step so device placement is correct and deterministic. This should make the fallback path run end-to-end and produce a valid `submission.csv`, and it should also improve your score versus the global-prior baseline because the kNN retrieval actually execute.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd

import fastai
from fastai.vision.all import *
import dill

import torchvision

pd.set_option("display.max_columns", 200)



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

test["image_path"] = test["image"].apply(lambda x: str(test_img_dir / x))

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
def extract_embeddings(image_paths, model, tfm, bs=64, device=None):
    """
    Bugfix: the original code applied `tfm(im)` which triggers fastai's Normalize on CPU
    tensors (mean/std) while image tensors may be moved to CUDA later. This can crash inside
    Normalize with a device mismatch.

    Minimal fix: apply the pipeline transforms step-by-step, moving tensors to `device`
    before Normalize, and moving Normalize.mean/std to `device` before use.
    Core embedding + normalization logic is unchanged.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    if isinstance(tfm, Pipeline):
        tfms = list(tfm.fs)
    else:
        tfms = [tfm]

    norm_tfm = None
    pre_tfms = []
    post_tfms = []
    found_norm = False
    for f in tfms:
        if (not found_norm) and isinstance(f, Normalize):
            norm_tfm = f
            found_norm = True
        elif not found_norm:
            pre_tfms.append(f)
        else:
            post_tfms.append(f)

    def apply_tfms(x, fs):
        for f in fs:
            x = f(x)
        return x

    embs = []
    n = len(image_paths)
    for i in range(0, n, bs):
        batch_paths = image_paths[i : i + bs]
        imgs = []
        for p in batch_paths:
            im = PILImage.create(p)
            if getattr(im, "mode", None) != "RGB":
                im = im.convert("RGB")

            x = apply_tfms(im, pre_tfms)

            if isinstance(x, torch.Tensor):
                x = x.to(device)

            if norm_tfm is not None and isinstance(x, torch.Tensor):
                if (
                    isinstance(norm_tfm.mean, torch.Tensor)
                    and norm_tfm.mean.device != x.device
                ):
                    norm_tfm.mean = norm_tfm.mean.to(x.device)
                if (
                    isinstance(norm_tfm.std, torch.Tensor)
                    and norm_tfm.std.device != x.device
                ):
                    norm_tfm.std = norm_tfm.std.to(x.device)
                x = norm_tfm(x)

            x = apply_tfms(x, post_tfms)

            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            x = x.to(device)

            imgs.append(x)

        x = torch.stack(imgs, dim=0)

        e = model(x)
        if e.ndim > 2:
            e = torch.flatten(e, 1)
        e = torch.nn.functional.normalize(e, p=2, dim=1)
        embs.append(e.detach().cpu())
    return torch.cat(embs, dim=0)


def knn_predict_top5(train_embs, train_labels, test_embs, k=50, topn=5):
    sims = test_embs @ train_embs.T  # [n_test, n_train]
    vals, idx = torch.topk(
        sims, k=min(k, train_embs.shape[0]), dim=1, largest=True, sorted=True
    )
    out = []
    for row_idx in idx.numpy():
        scores = {}
        for j in row_idx:
            hid = train_labels[int(j)]
            scores[hid] = scores.get(hid, 0.0) + 1.0
        top = sorted(scores.items(), key=lambda x: (-x[1], str(x[0])))[:topn]
        out.append(" ".join([str(t[0]) for t in top]))
    return out


def _top5_global_prior(train_csv_path):
    tr = pd.read_csv(train_csv_path, usecols=["hotel_id"])
    top5 = tr["hotel_id"].astype(str).value_counts().head(5).index.tolist()
    while len(top5) < 5:
        top5.append(top5[-1] if top5 else "0")
    return top5[:5]


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
    tr["image_path"] = tr.apply(_build_image_path_train, axis=1)

    exists_mask = tr["image_path"].apply(lambda p: Path(p).exists())
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

    tfm = Pipeline(
        [
            Resize(224),
            ToTensor(),
            IntToFloatTensor(),
            Normalize.from_stats(*imagenet_stats),
        ]
    )

    train_paths = tr["image_path"].tolist()
    train_labels = tr["hotel_id"].astype(str).tolist()
    test_paths = test_df["image_path"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device).eval()

    train_embs = extract_embeddings(train_paths, backbone, tfm, bs=64, device=device)
    test_embs = extract_embeddings(test_paths, backbone, tfm, bs=64, device=device)

    preds = knn_predict_top5(train_embs, train_labels, test_embs, k=50, topn=5)
    preds = _ensure_5_ids(preds, _top5_global_prior(train_csv_path))
    return preds




## === cell 6
preds = None

if use_models:
    probs = None
    last_learn = None

    for model_path in models:
        cpu_flag = not torch.cuda.is_available()
        learn = load_learner(fname=Path(model_path), cpu=cpu_flag, pickle_module=dill)
        last_learn = learn

        test_dl = learn.dls.test_dl(
            test,
            with_labels=False,
            item_tfms=[Transform(lambda o: PILImage.create(o["image_path"]))],
        )

        probs_temp, _ = learn.tta(dl=test_dl, n=6)

        if probs is None:
            probs = probs_temp
        else:
            probs += probs_temp

    probs = probs / len(models)
    preds = make_preds_from_probs(probs, last_learn.dls.vocab)
else:
    preds = fallback_knn_submission(test, train_csv, max_train_images=60000, seed=0)

preds[:3], len(preds)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1755298451.py in <cell line: 0>()
     26     preds = make_preds_from_probs(probs, last_learn.dls.vocab)
     27 else:
---> 28     preds = fallback_knn_submission(test, train_csv, max_train_images=60000, seed=0)
     29 
     30 preds[:3], len(preds)

/tmp/ipykernel_55/3469803773.py in fallback_knn_submission(test_df, train_csv_path, max_train_images, seed)
    167     backbone = backbone.to(device).eval()
    168 
--> 169     train_embs = extract_embeddings(train_paths, backbone, tfm, bs=64, device=device)
    170     test_embs = extract_embeddings(test_paths, backbone, tfm, bs=64, device=device)
    171 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/3469803773.py in extract_embeddings(image_paths, model, tfm, bs, device)
     83         x = torch.stack(imgs, dim=0)
     84 
---> 85         e = model(x)
     86         if e.ndim > 2:
     87             e = torch.flatten(e, 1)

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

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

RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [64, 1, 3, 224, 224]

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
