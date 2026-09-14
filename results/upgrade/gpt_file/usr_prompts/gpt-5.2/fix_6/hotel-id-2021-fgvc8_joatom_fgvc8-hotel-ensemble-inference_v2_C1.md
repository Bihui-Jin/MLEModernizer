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

import pandas as pd
import numpy as np

import fastai
from fastai.vision.all import *

import dill



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
models = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",
    "../input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",
]



## === cell 3
INPUT_CANDIDATES = [
    Path("../input/hotel-id-2021-fgvc8"),
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("../input"),  # fallback if the dataset is mounted directly under input
    Path("/kaggle/input"),
]


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


base_input = _first_existing(INPUT_CANDIDATES[:2]) or _first_existing(
    INPUT_CANDIDATES[2:]
)
base_input



## === cell 4
test_images_candidates = [
    base_input / "test_images",
    Path("../input/hotel-id-2021-fgvc8/test_images"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/test_images"),
    Path("../input/test_images"),
    Path("/kaggle/input/test_images"),
]
test_images_dir = _first_existing(test_images_candidates)
if test_images_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {test_images_candidates}"
    )

sample_sub_candidates = [
    base_input / "sample_submission.csv",
    Path("../input/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"),
    Path("../input/sample_submission.csv"),
    Path("/kaggle/input/sample_submission.csv"),
]
sample_sub_path = _first_existing(sample_sub_candidates)
if sample_sub_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv. Tried: {sample_sub_candidates}"
    )

submission = pd.read_csv(sample_sub_path)
test = submission.copy()

test["image_path"] = test["image"].apply(lambda x: str((test_images_dir / x).resolve()))
missing = (
    test.loc[~test["image_path"].map(lambda p: Path(p).exists()), "image"]
    .head(10)
    .tolist()
)
if len(missing) > 0:
    print("Warning: some test image paths do not exist (showing up to 10):", missing)

test.head()




## === cell 5
def _resolve_model_paths(requested_paths):
    resolved = []
    missing = []
    for p in requested_paths:
        pp = Path(p)
        if pp.exists():
            resolved.append(pp)
        else:
            missing.append(p)

    search_roots = [Path("../input"), Path("/kaggle/input")]
    found = []
    for root in search_roots:
        if root.exists():
            try:
                found.extend(list(root.rglob("*.pkl")))
            except Exception:
                pass

    def _rank(p: Path):
        name = p.name.lower()
        s = str(p).lower()
        export_bonus = 0 if ("export" in name or "learner" in name) else 2
        comp_bonus = 0 if ("hotel" in s or "fgvc" in s or "hotel-id-2021" in s) else 1
        arch_bonus = (
            0
            if any(
                k in name
                for k in ["dn161", "densenet", "res101", "resnet", "eff", "vit"]
            )
            else 1
        )
        length_penalty = len(s) / 1000.0
        return (export_bonus, comp_bonus, arch_bonus, length_penalty)

    found_sorted = sorted(found, key=_rank)

    if len(resolved) == 0:
        resolved = found_sorted[:3]

    return resolved, missing, found_sorted


resolved_models, missing_models, discovered_models = _resolve_model_paths(models)

print("Requested models:", models)
print("Resolved models:", [str(p) for p in resolved_models])
if missing_models:
    print("Missing requested models:", missing_models)
print("Discovered *.pkl count:", len(discovered_models))

probs2 = None
learn = None
loaded_any = False

for model_path in resolved_models:
    try:
        learn_tmp = load_learner(fname=Path(model_path), cpu=False, pickle_module=dill)

        test_dl = learn_tmp.dls.test_dl(
            test, with_labels=False, item_tfms=None, rm_type_tfms=None
        )
        try:
            probs_temp, _ = learn_tmp.tta(dl=test_dl)
        except Exception:
            test_dl2 = learn_tmp.dls.test_dl(test["image_path"].tolist())
            probs_temp, _ = learn_tmp.tta(dl=test_dl2)

        probs_temp = torch.clamp(probs_temp, min=1e-12)

        if probs2 is None:
            probs2 = torch.log(probs_temp)
        else:
            probs2 += torch.log(probs_temp)

        learn = learn_tmp  # keep the last successfully loaded learner for vocab
        loaded_any = True
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Skipping model {model_path} due to error: {repr(e)}")

print("Loaded any models:", loaded_any)



## === cell 6
if loaded_any and probs2 is not None:
    preds_idx = probs2.topk(5, dim=1)[1].cpu().numpy()
else:
    preds_idx = None

preds_idx is None, (None if preds_idx is None else preds_idx.shape)



## === cell 7
if preds_idx is not None:
    vocab = np.array(learn.dls.vocab)
    preds = [" ".join(map(str, vocab[row])) for row in preds_idx]
else:
    train_csv_candidates = [
        base_input / "train.csv",
        Path("../input/hotel-id-2021-fgvc8/train.csv"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv"),
        Path("../input/train.csv"),
        Path("/kaggle/input/train.csv"),
    ]
    train_path = _first_existing(train_csv_candidates)
    if train_path is None:
        raise FileNotFoundError(
            f"Could not find train.csv. Tried: {train_csv_candidates}"
        )

    train_df = pd.read_csv(train_path)

    train_images_candidates = [
        base_input / "train_images",
        Path("../input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images"),
        Path("../input/train_images"),
        Path("/kaggle/input/train_images"),
    ]
    train_images_dir = _first_existing(train_images_candidates)
    if train_images_dir is None:
        raise FileNotFoundError(
            f"Could not find train_images directory. Tried: {train_images_candidates}"
        )

    train_df["image_path"] = train_df.apply(
        lambda r: str((train_images_dir / str(r["chain"]) / r["image"]).resolve()),
        axis=1,
    )

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader("image_path"),
        get_y=ColReader("hotel_id"),
        splitter=RandomSplitter(valid_pct=0.0, seed=42),
        item_tfms=Resize(224, method="squish"),
        batch_tfms=Normalize.from_stats(*imagenet_stats),
    )

    dls = dblock.dataloaders(
        train_df[["image_path", "hotel_id"]].copy(), bs=64, num_workers=2
    )
    learn_emb = vision_learner(dls, resnet50, pretrained=True, metrics=None)
    learn_emb.model.eval()

    emb_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    learn_emb.model.to(emb_device)

    def _get_embeds(paths, bs=64):
        dl = learn_emb.dls.test_dl(paths, bs=bs, num_workers=2)
        with torch.no_grad():
            feats = learn_emb.get_preds(dl=dl, with_decoded=False)[0]
        return feats

    max_train_ref = 30000
    train_ref = train_df.copy()
    if len(train_ref) > max_train_ref:
        train_ref = train_ref.sample(n=max_train_ref, random_state=42).reset_index(
            drop=True
        )

    train_paths = train_ref["image_path"].tolist()
    test_paths = test["image_path"].tolist()

    train_emb = _get_embeds(train_paths, bs=64).float().to("cpu")
    test_emb = _get_embeds(test_paths, bs=64).float().to("cpu")

    train_emb = train_emb / (train_emb.norm(dim=1, keepdim=True) + 1e-12)
    test_emb = test_emb / (test_emb.norm(dim=1, keepdim=True) + 1e-12)

    k = 5
    hotel_ids = train_ref["hotel_id"].astype(str).values
    preds = []
    chunk = 512
    train_emb_t = train_emb.t()
    for i in range(0, test_emb.shape[0], chunk):
        te = test_emb[i : i + chunk]
        sims = te @ train_emb_t
        topk_idx = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
        for row in topk_idx:
            preds.append(" ".join(hotel_ids[row].tolist()))

if len(preds) != len(submission):
    raise RuntimeError(
        f"Predictions length {len(preds)} does not match submission length {len(submission)}"
    )

preds[:3], len(preds)



## === cell 8
submission_out = submission.copy()
submission_out["hotel_id"] = preds

submission_out["image"] = submission_out["image"].astype(str)
submission_out["hotel_id"] = submission_out["hotel_id"].astype(str)

submission_out.to_csv("submission.csv", index=False)
submission_out.head()
