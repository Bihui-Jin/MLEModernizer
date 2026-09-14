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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



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

test["image_path"] = (
    test_images_dir.resolve().as_posix() + "/" + test["image"].astype(str)
)

missing = []
for img, p in zip(
    test["image"].head(10).tolist(), test["image_path"].head(10).tolist()
):
    if not Path(p).exists():
        missing.append(img)
if len(missing) > 0:
    print("Warning: some test image paths do not exist (showing up to 10):", missing)

test.head()




## === cell 5
def _resolve_model_paths(requested_paths):
    """
    Prefer true exported fastai Learners (usually 'export*.pkl').
    If none are present in the environment, we will later fall back.
    """
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

        export_bonus = (
            0 if name.startswith("export") else (1 if "export" in name else 5)
        )
        comp_bonus = 0 if ("hotel-id-2021" in s or "fgvc" in s or "hotel" in s) else 3
        arch_bonus = (
            0
            if any(k in name for k in ["dn", "densenet", "res", "resnet", "eff", "vit"])
            else 2
        )
        depth_penalty = s.count("/") / 60.0
        return (export_bonus, comp_bonus, arch_bonus, depth_penalty)

    found_sorted = sorted(found, key=_rank)

    if len(resolved) == 0:
        resolved = found_sorted[:40]

    return resolved, missing, found_sorted


resolved_models, missing_models, discovered_models = _resolve_model_paths(models)

print("Requested models:", models)
print(
    "Resolved models (candidates to try):",
    [str(p) for p in resolved_models[:10]],
    "...",
)
if missing_models:
    print("Missing requested models:", missing_models)
print("Discovered *.pkl count:", len(discovered_models))

_CPU_COUNT = os.cpu_count() or 2
_NW = min(8, max(2, _CPU_COUNT // 2))
_PIN = torch.cuda.is_available()



## === cell 6
train_csv_candidates = [
    base_input / "train.csv",
    Path("../input/hotel-id-2021-fgvc8/train.csv"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv"),
    Path("../input/train.csv"),
    Path("/kaggle/input/train.csv"),
]
train_path = _first_existing(train_csv_candidates)
if train_path is None:
    raise FileNotFoundError(f"Could not find train.csv. Tried: {train_csv_candidates}")

train_df_for_vocab = pd.read_csv(train_path, usecols=["hotel_id"])
train_hotel_ids_raw = train_df_for_vocab["hotel_id"].astype(str).tolist()
del train_df_for_vocab


def _norm_hid(x: str) -> str:
    x = str(x).strip()
    if x.isdigit():
        x2 = x.lstrip("0")
        return x2 if x2 != "" else "0"
    return x


train_hotel_ids = set(_norm_hid(x) for x in train_hotel_ids_raw)
train_hotel_ids_strict = set(str(x).strip() for x in train_hotel_ids_raw)


def _extract_vocab(learn_tmp):
    v = getattr(learn_tmp.dls, "vocab", None)
    if v is None:
        return None
    try:
        v_list = list(map(str, list(v)))
    except Exception:
        try:
            v_list = list(map(str, v.items))
        except Exception:
            return None
    return v_list


def _vocab_overlap_ratio(vocab_list, hotel_id_set, max_check=8000, normalize=False):
    if vocab_list is None or len(vocab_list) == 0:
        return 0.0
    vv = vocab_list[:max_check] if len(vocab_list) > max_check else vocab_list
    if normalize:
        hit = sum((_norm_hid(x) in hotel_id_set) for x in vv)
    else:
        hit = sum((str(x).strip() in hotel_id_set) for x in vv)
    return hit / float(len(vv))


def _is_reasonable_hotel_vocab(vocab_list):
    if vocab_list is None or len(vocab_list) < 100:
        return False, 0.0, 0.0
    overlap_strict = _vocab_overlap_ratio(
        vocab_list, train_hotel_ids_strict, normalize=False
    )
    overlap_norm = _vocab_overlap_ratio(vocab_list, train_hotel_ids, normalize=True)
    ok = (overlap_strict >= 0.25) or (overlap_norm >= 0.35)
    return ok, overlap_strict, overlap_norm


accum_logprob = None  # torch.Tensor [n_test, n_classes_union]
union_hotel_ids = []  # list[str] union vocab (string hotel IDs)
union_index = {}  # hotel_id -> col idx

loaded_any = False
loaded_count = 0

test_paths = test["image_path"].tolist()

accepted_models = []
rejected_models = []

for model_path in resolved_models:
    try:
        learn_tmp = load_learner(fname=Path(model_path), cpu=False, pickle_module=dill)

        vocab_list = _extract_vocab(learn_tmp)
        ok_vocab, ov_strict, ov_norm = _is_reasonable_hotel_vocab(vocab_list)
        if not ok_vocab:
            raise ValueError(
                f"Vocab does not look like hotel_id classes (overlap_strict={ov_strict:.3f}, overlap_norm={ov_norm:.3f}, vocab_len={0 if vocab_list is None else len(vocab_list)})"
            )

        test_dl = learn_tmp.dls.test_dl(
            test_paths,
            with_labels=False,
            num_workers=_NW,
            pin_memory=_PIN,
            persistent_workers=(_NW > 0),
        )

        with torch.inference_mode():
            try:
                probs_temp, _ = learn_tmp.tta(dl=test_dl)
            except Exception:
                probs_temp, _ = learn_tmp.tta(dl=test_dl, n=0)

        probs_temp = torch.clamp(probs_temp, min=1e-12)
        logp = torch.log(probs_temp)  # [n_test, n_vocab_model]

        model_vocab = [_norm_hid(x) for x in vocab_list]

        if accum_logprob is None:
            union_hotel_ids = model_vocab
            union_index = {h: i for i, h in enumerate(union_hotel_ids)}
            accum_logprob = logp
        else:
            new_ids = [h for h in model_vocab if h not in union_index]
            if len(new_ids) > 0:
                for h in new_ids:
                    union_index[h] = len(union_hotel_ids)
                    union_hotel_ids.append(h)
                pad = torch.zeros(
                    (accum_logprob.shape[0], len(new_ids)),
                    device=accum_logprob.device,
                    dtype=accum_logprob.dtype,
                )
                accum_logprob = torch.cat([accum_logprob, pad], dim=1)

            cols = torch.tensor(
                [union_index[h] for h in model_vocab],
                device=accum_logprob.device,
                dtype=torch.long,
            )
            accum_logprob[:, cols] += logp

        loaded_any = True
        loaded_count += 1
        accepted_models.append((str(model_path), ov_strict, ov_norm, len(vocab_list)))
        print(
            f"Accepted model: {model_path} (overlap_strict={ov_strict:.3f}, overlap_norm={ov_norm:.3f}, vocab_len={len(vocab_list)})"
        )
    except Exception as e:
        rejected_models.append((str(model_path), repr(e)))
        print(f"Rejected model {model_path} due to: {repr(e)}")

if loaded_any and accum_logprob is not None and loaded_count > 0:
    accum_logprob = accum_logprob / float(loaded_count)

print(
    "Accepted models:",
    accepted_models[:10],
    ("..." if len(accepted_models) > 10 else ""),
)
print("Rejected models count:", len(rejected_models))
print("Loaded any models:", loaded_any, "count:", loaded_count)



## === cell 7
if loaded_any and accum_logprob is not None:
    preds_idx = accum_logprob.topk(5, dim=1)[1].cpu().numpy()
else:
    preds_idx = None

preds_idx is None, (None if preds_idx is None else preds_idx.shape)




## === cell 8
def _dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


preds = None

if preds_idx is not None:
    vocab_union = np.array(union_hotel_ids, dtype=object)
    preds_list = []
    for row in preds_idx:
        ids = vocab_union[row].tolist()
        ids = _dedupe_keep_order(ids)
        if len(ids) < 5:
            ids = ids + ids[: (5 - len(ids))]
        preds_list.append(" ".join(ids[:5]))
    preds = preds_list
else:
    try:
        train_meta = pd.read_csv(train_path, usecols=["image", "chain", "hotel_id"])
        train_meta["image"] = train_meta["image"].astype(str)
        train_meta["chain"] = train_meta["chain"].astype(str)
        train_meta["hotel_id"] = train_meta["hotel_id"].astype(str).map(_norm_hid)

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

        train_meta["image_path"] = (
            train_images_dir.resolve().as_posix()
            + "/"
            + train_meta["chain"].astype(str)
            + "/"
            + train_meta["image"].astype(str)
        )

        exists_mask = train_meta["image_path"].map(lambda p: Path(p).exists())
        if exists_mask.mean() < 0.9:
            alt_paths = (
                train_images_dir.resolve().as_posix()
                + "/"
                + train_meta["image"].astype(str)
            )
            alt_exists = alt_paths.map(lambda p: Path(p).exists())
            use_alt = alt_exists.sum() > exists_mask.sum()
            if use_alt:
                train_meta["image_path"] = alt_paths
                exists_mask = alt_exists

        train_meta = train_meta.loc[exists_mask].reset_index(drop=True)

        if len(train_meta) < 1000:
            train_full = pd.read_csv(train_path, usecols=["hotel_id"])
            top5 = (
                train_full["hotel_id"]
                .astype(str)
                .map(_norm_hid)
                .value_counts()
                .head(5)
                .index.tolist()
            )
            del train_full
            top5 = _dedupe_keep_order(top5)
            if len(top5) < 5:
                top5 = (top5 + top5 * 5)[:5]
            fallback_pred = " ".join(top5[:5])
            preds = [fallback_pred] * len(submission)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            base_model = resnet18(weights="DEFAULT")
            body = create_body(base_model, pretrained=False, cut=-2)

            emb_model = (
                nn.Sequential(body, AdaptiveConcatPool2d(), Flatten()).to(device).eval()
            )

            item_tfms = [Resize(224)]
            batch_tfms = [Normalize.from_stats(*imagenet_stats)]
            dblock = DataBlock(
                blocks=(ImageBlock,),
                get_x=ColReader("image_path"),
                splitter=RandomSplitter(valid_pct=0.0, seed=42),
                item_tfms=item_tfms,
                batch_tfms=batch_tfms,
            )

            bs = 64 if torch.cuda.is_available() else 32
            train_dl = dblock.dataloaders(
                train_meta, bs=bs, shuffle_train=False, num_workers=_NW, pin_memory=_PIN
            ).train
            test_df_for_dl = test[["image_path"]].copy()
            test_dl = dblock.dataloaders(
                test_df_for_dl,
                bs=bs,
                shuffle_train=False,
                num_workers=_NW,
                pin_memory=_PIN,
            ).train

            def _extract_embs(dl):
                embs = []
                with torch.inference_mode():
                    for batch in dl:
                        xb = batch[0].to(device, non_blocking=True)
                        out = emb_model(xb)
                        out = F.normalize(out, p=2, dim=1)
                        embs.append(out.detach().cpu())
                return torch.cat(embs, dim=0)

            train_emb = _extract_embs(train_dl)  # [n_train, d]
            test_emb = _extract_embs(test_dl)  # [n_test, d]

            max_train = 30000
            if train_emb.shape[0] > max_train:
                freq = train_meta["hotel_id"].value_counts()
                keep_hotels = freq.index.tolist()
                keep_rows = []
                per_hotel_cap = 6
                counts = {}
                for i, hid in enumerate(train_meta["hotel_id"].tolist()):
                    c = counts.get(hid, 0)
                    if c < per_hotel_cap:
                        keep_rows.append(i)
                        counts[hid] = c + 1
                    if len(keep_rows) >= max_train:
                        break
                keep_rows = np.array(keep_rows, dtype=np.int64)
                train_meta_small = train_meta.iloc[keep_rows].reset_index(drop=True)
                train_emb_small = train_emb[keep_rows]
            else:
                train_meta_small = train_meta
                train_emb_small = train_emb

            K = 50
            chunk = 256
            train_emb_small_t = train_emb_small.t().contiguous()
            hotel_ids_train = train_meta_small["hotel_id"].tolist()

            preds_list = []
            for i in range(0, test_emb.shape[0], chunk):
                te = test_emb[i : i + chunk]  # [c, d]
                sims = te @ train_emb_small_t  # [c, n_train_small]
                topk_idx = (
                    torch.topk(sims, k=min(K, sims.shape[1]), dim=1)
                    .indices.cpu()
                    .numpy()
                )
                for row in topk_idx:
                    votes = {}
                    for rank, idx in enumerate(row.tolist(), start=1):
                        hid = hotel_ids_train[idx]
                        votes[hid] = votes.get(hid, 0.0) + 1.0 / rank
                    ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
                    top_ids = [hid for hid, _ in ranked[:5]]
                    top_ids = _dedupe_keep_order(top_ids)
                    if len(top_ids) < 5:
                        pad_ids = (
                            train_meta_small["hotel_id"]
                            .value_counts()
                            .index.tolist()[:10]
                        )
                        for hid in pad_ids:
                            if len(top_ids) >= 5:
                                break
                            if hid not in top_ids:
                                top_ids.append(hid)
                    preds_list.append(" ".join(top_ids[:5]))

            preds = preds_list

        if len(preds) != len(submission):
            raise RuntimeError(
                f"Predictions length {len(preds)} does not match submission length {len(submission)}"
            )
    except Exception as e:
        print("Embedding fallback failed; using frequency fallback due to:", repr(e))
        train_full = pd.read_csv(train_path, usecols=["hotel_id"])
        top5 = (
            train_full["hotel_id"]
            .astype(str)
            .map(_norm_hid)
            .value_counts()
            .head(5)
            .index.tolist()
        )
        del train_full
        top5 = _dedupe_keep_order(top5)
        if len(top5) < 5:
            top5 = (top5 + top5 * 5)[:5]
        fallback_pred = " ".join(top5[:5])
        preds = [fallback_pred] * len(submission)

preds[:3], len(preds)



## === cell 9
submission_out = submission.copy()
submission_out["hotel_id"] = preds

submission_out["image"] = submission_out["image"].astype(str)
submission_out["hotel_id"] = submission_out["hotel_id"].astype(str)

submission_out.to_csv("submission.csv", index=False)
submission_out.head()
