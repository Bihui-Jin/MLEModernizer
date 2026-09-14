# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5399174875810392

# 6. Current score

0.00469

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I fix the missing model files by switching to a robust path discovery (searching under `../input` and `/kaggle/input`) and gracefully skipping any models that still can’t be found, so the notebook runs end-to-end. I also fix the ensemble accumulation bug (`probs` was never updated, so later models weren’t actually combined) while preserving the same log-prob averaging approach. To ensure a valid submission is always produced, I add a safe fallback that fills predictions with the most frequent training `hotel_id` values if no models are loadable. Finally, I keep the submission format exactly as required (`image,hotel_id`) and write `submission.csv` to the working directory.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.5399), so we need a real accuracy lift with minimal disruption to your pipeline. The biggest issue is that your code likely isn’t loading the intended pretrained models at all (or is loading mismatched ones), so it falls back to “top5 most frequent” predictions which score near-zero on MAP@5. I keep your exact inference/ensemble logic, but make model discovery robust to this competition’s dataset layout by searching specifically under `../input` for any `.pkl` exports and preferring files that look like hotel-id exports, and I fail loudly (with a controlled message) if we still load zero models so you don’t unknowingly submit a fallback. I also fix the test image path to match the actual provided structure and ensure the test DataLoader is built from file paths that fastai can open reliably.'
- What this solution (achieved 0.00469) has done: 'The timeout is dominated by running `learn.tta()` for each model, because TTA multiplies inference cost by the number of augmentations and also rebuilds a test DataLoader per model. To keep identical prediction semantics while speeding up, the script below (a) builds each model’s test DataLoader once (avoids Pandas-row overhead and repeated transform/type checks), (b) runs TTA with fastai’s “use max version” behavior unchanged but reduces Python overhead by using `torch.inference_mode()` and keeping tensors on-device until the final top-k, and (c) sets efficient DataLoader settings (`num_workers` based on CPU count, `pin_memory`, `persistent_workers`) that don’t change outputs. The fallback embedding-based block is preserved (same logic) but optimized to avoid slow `df.apply(axis=1)` and to vectorize path building; it only runs if no .pkl loads successfully.'
- What this solution (achieved 0.00469) has done: 'Your current score (0.00469) is far below the target (0.5399), so we need a real accuracy lift with minimal disruption; the biggest likely issue is that your ensemble is often not actually using the intended pretrained hotel-ID models and/or is using the wrong class vocabulary mapping. I keep your exact “load exported fastai learners + TTA + log-prob sum + top5” core logic, but make model discovery stricter (prefer actual hotel-id export learners and require at least one to load), and I also enforce correct mapping from prediction indices to hotel_id by using the learner’s `dls.vocab` (and ensuring it’s the Category vocab, not an unrelated one). Finally, I keep the fallback path, but only use it if truly necessary, and I fix a common subtlety: averaging log-probs by number of models (doesn’t change ranking if all models contribute, but prevents numeric skew if you later change model count and helps stability).'

# 9. Code solution

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
    Change rationale (score): the prior run likely loaded zero/irrelevant .pkl files,
    causing near-random/fallback predictions. We keep the same ensemble logic but
    make discovery prefer real exported learners for this competition and ensure we
    actually load at least one usable model (otherwise fallback is used explicitly).
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

        export_bonus = 0 if ("export" in name or "learner" in name) else 5
        comp_bonus = 0 if ("hotel" in s or "fgvc" in s or "hotel-id-2021" in s) else 3

        arch_bonus = (
            0
            if any(
                k in name
                for k in ["dn161", "densenet", "res101", "resnet", "eff", "vit"]
            )
            else 2
        )

        length_penalty = len(s) / 2000.0
        return (export_bonus, comp_bonus, arch_bonus, length_penalty)

    found_sorted = sorted(found, key=_rank)

    if len(resolved) == 0:
        resolved = found_sorted[:5]

    return resolved, missing, found_sorted


resolved_models, missing_models, discovered_models = _resolve_model_paths(models)

print("Requested models:", models)
print("Resolved models:", [str(p) for p in resolved_models])
if missing_models:
    print("Missing requested models:", missing_models)
print("Discovered *.pkl count:", len(discovered_models))

_CPU_COUNT = os.cpu_count() or 2
_NW = min(8, max(2, _CPU_COUNT // 2))
_PIN = torch.cuda.is_available()

probs2 = None
learn = None
loaded_any = False
loaded_count = 0

test_paths = test["image_path"].tolist()

for model_path in resolved_models:
    try:
        learn_tmp = load_learner(fname=Path(model_path), cpu=False, pickle_module=dill)

        vocab_tmp = getattr(learn_tmp.dls, "vocab", None)
        if vocab_tmp is None or len(vocab_tmp) < 10:
            raise ValueError(
                f"Suspicious/empty vocab in learner: {type(vocab_tmp)} len={0 if vocab_tmp is None else len(vocab_tmp)}"
            )

        test_dl = learn_tmp.dls.test_dl(
            test_paths,
            with_labels=False,
            num_workers=_NW,
            pin_memory=_PIN,
            persistent_workers=(_NW > 0),
        )

        with torch.inference_mode():
            probs_temp, _ = learn_tmp.tta(dl=test_dl)

        probs_temp = torch.clamp(probs_temp, min=1e-12)

        if probs2 is None:
            probs2 = torch.log(probs_temp)
        else:
            probs2 += torch.log(probs_temp)

        learn = learn_tmp  # keep last good learner for vocab
        loaded_any = True
        loaded_count += 1
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Skipping model {model_path} due to error: {repr(e)}")

if loaded_any and probs2 is not None and loaded_count > 0:
    probs2 = probs2 / float(loaded_count)

print("Loaded any models:", loaded_any, "count:", loaded_count)



## === cell 6
if loaded_any and probs2 is not None:
    preds_idx = probs2.topk(5, dim=1)[1].cpu().numpy()
else:
    preds_idx = None

preds_idx is None, (None if preds_idx is None else preds_idx.shape)



## === cell 7
if preds_idx is not None:
    vocab = np.array(list(map(str, learn.dls.vocab)))
    preds = [" ".join(vocab[row].tolist()) for row in preds_idx]
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

    base_train = train_images_dir.resolve().as_posix()
    train_df["image_path"] = (
        base_train
        + "/"
        + train_df["chain"].astype(str)
        + "/"
        + train_df["image"].astype(str)
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
        train_df[["image_path", "hotel_id"]].copy(),
        bs=64,
        num_workers=_NW,
        pin_memory=_PIN,
        persistent_workers=(_NW > 0),
    )
    learn_emb = vision_learner(dls, resnet50, pretrained=True, metrics=None)
    learn_emb.model.eval()

    emb_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    learn_emb.model.to(emb_device)

    def _get_embeds(paths, bs=64):
        dl = learn_emb.dls.test_dl(
            paths, bs=bs, num_workers=_NW, pin_memory=_PIN, persistent_workers=(_NW > 0)
        )
        with torch.inference_mode():
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
