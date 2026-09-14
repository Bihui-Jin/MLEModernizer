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

0.5399174875810392

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I fix the missing model files by switching to a robust path discovery (searching under `../input` and `/kaggle/input`) and gracefully skipping any models that still can’t be found, so the notebook runs end-to-end. I also fix the ensemble accumulation bug (`probs` was never updated, so later models weren’t actually combined) while preserving the same log-prob averaging approach. To ensure a valid submission is always produced, I add a safe fallback that fills predictions with the most frequent training `hotel_id` values if no models are loadable. Finally, I keep the submission format exactly as required (`image,hotel_id`) and write `submission.csv` to the working directory.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.5399), so we need a real accuracy lift with minimal disruption to your pipeline. The biggest issue is that your code likely isn’t loading the intended pretrained models at all (or is loading mismatched ones), so it falls back to “top5 most frequent” predictions which score near-zero on MAP@5. I keep your exact inference/ensemble logic, but make model discovery robust to this competition’s dataset layout by searching specifically under `../input` for any `.pkl` exports and preferring files that look like hotel-id exports, and I fail loudly (with a controlled message) if we still load zero models so you don’t unknowingly submit a fallback. I also fix the test image path to match the actual provided structure and ensure the test DataLoader is built from file paths that fastai can open reliably.'
- What this solution (achieved 0.00469) has done: 'The timeout is dominated by running `learn.tta()` for each model, because TTA multiplies inference cost by the number of augmentations and also rebuilds a test DataLoader per model. To keep identical prediction semantics while speeding up, the script below (a) builds each model’s test DataLoader once (avoids Pandas-row overhead and repeated transform/type checks), (b) runs TTA with fastai’s “use max version” behavior unchanged but reduces Python overhead by using `torch.inference_mode()` and keeping tensors on-device until the final top-k, and (c) sets efficient DataLoader settings (`num_workers` based on CPU count, `pin_memory`, `persistent_workers`) that don’t change outputs. The fallback embedding-based block is preserved (same logic) but optimized to avoid slow `df.apply(axis=1)` and to vectorize path building; it only runs if no .pkl loads successfully.'
- What this solution (achieved 0.00469) has done: 'Your current score (0.00469) is far below the target (0.5399), so we need a real accuracy lift with minimal disruption; the biggest likely issue is that your ensemble is often not actually using the intended pretrained hotel-ID models and/or is using the wrong class vocabulary mapping. I keep your exact “load exported fastai learners + TTA + log-prob sum + top5” core logic, but make model discovery stricter (prefer actual hotel-id export learners and require at least one to load), and I also enforce correct mapping from prediction indices to hotel_id by using the learner’s `dls.vocab` (and ensuring it’s the Category vocab, not an unrelated one). Finally, I keep the fallback path, but only use it if truly necessary, and I fix a common subtlety: averaging log-probs by number of models (doesn’t change ranking if all models contribute, but prevents numeric skew if you later change model count and helps stability).'
- What this solution (achieved 0.00469) has done: 'Your score is far below the target, so the smallest meaningful way to move toward it (without changing your model/ensemble logic) is to ensure we’re actually using the correct class-to-hotel_id mapping and not silently mixing in learners whose `dls.vocab` is unrelated to `hotel_id`. I add a strict vocab validation against `train.csv` hotel IDs and only ensemble models whose vocab overlaps strongly with true hotel IDs (this keeps the same TTA + log-prob averaging + top5 core logic, but prevents “wrong vocab = wrong labels” submissions). I also ensure we use `learn.dls.vocab` as the authoritative mapping for each model and reject any model where the vocab length or overlap is suspicious. Finally, I keep your fallback path intact, but it should trigger far less often once the correct learners are selected.'
- What this solution (achieved 0.00469) has done: 'Your score is far below the target, so the most likely blocker is that you are still not ensembling real pretrained hotel-ID classifiers (your hardcoded model paths don’t exist, and the strict “overlap>=0.90” filter can reject everything), which forces the weak fallback and yields ~0 MAP@5. I keep your exact “load fastai exported learners → build test_dl → TTA → log-prob average → top-5” core logic, but (1) tighten model discovery to prefer *valid* exported Learners and load more candidates, and (2) relax the vocab validation just enough to accept legitimate learners whose vocab is `Category`/`L` but still matches hotel IDs (while still rejecting clearly-wrong vocabs). Finally, I fix a subtle mapping bug risk by using each model’s own vocab to convert its top-5 indices to hotel_ids, then ensemble by summed log-probs in a shared hotel_id space (this preserves your ensemble semantics but avoids “last learner vocab” mis-mapping when models differ). These are minimal, execution-safe changes aimed at moving you toward the target by ensuring correct labels and a non-fallback submission.'
- What this solution (achieved 0.00469) has done: 'Your score is far below the target, so the smallest meaningful improvement is to ensure the ensemble is actually using valid hotel-ID learners and that their vocab mapping is correct, rather than silently rejecting them and falling back. I keep your exact “load learner → build test_dl → TTA → log-prob sum/avg → top-5” logic, but relax the vocab filter in a safer way by normalizing hotel_id strings (strip/leading zeros) and accepting models whose vocab has reasonable overlap under normalization. I also fix an inference-shape pitfall by forcing `tta(n=0)` (no augmentation) only if the learner raises due to missing batch_tfms/tta incompatibility; otherwise keep current TTA behavior—this prevents total rejection due to TTA issues while preserving semantics when it works. Finally, I make the fallback embedding block always output unique top-5 hotel_ids per row (dedupe) to slightly improve MAP@5 versus repeated IDs, without changing the overall approach.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so the smallest high-impact change is to stop silently accepting unusable `.pkl` files and to ensure we actually load exported fastai Learners; if none load, we force a clear failure rather than submitting the weak fallback. Next, we keep your exact TTA + log-prob averaging + top-5 logic, but fix a key ensemble bug: when model vocabularies differ, you must add this model’s log-probs into the correct union columns (scatter-add), not `index_add_` into a newly appended matrix (which misaligns columns and destroys accuracy). Finally, we make the model search prefer `export*.pkl` and validate that the learner can build a test dataloader before accepting it, which should move MAP@5 substantially toward the target without changing model architecture/training semantics.'
- What this solution (achieved 0.00209) has done: 'We fix the hard failure caused by “no models loaded” by reinstating a safe, deterministic fallback that still produces a valid `submission.csv` end-to-end in this Kaggle environment. Since there are no attached/exported fastai `.pkl` models available in your provided file tree, we can’t legitimately reach the target MAP@5 via inference; the minimal correctness fix is to generate reasonable top-5 guesses using the most frequent `hotel_id`s from `train.csv` (plus uniqueness/deduping). I keep your existing ensemble/TTA logic unchanged when models are present, but if none load we not raise—just fall back and write the CSV in the required format. This move the score from a broken/0.0 pipeline to a small-but-nonzero baseline while preserving your core approach when real models are provided.'

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
    Prefer true exported fastai Learners (usually 'export*.pkl').
    If none are present in the environment, we will later fall back (in a stronger way than global-frequency).
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


if preds_idx is not None:
    vocab_union = np.array(union_hotel_ids, dtype=object)
    preds = []
    for row in preds_idx:
        ids = vocab_union[row].tolist()
        ids = _dedupe_keep_order(ids)
        if len(ids) < 5:
            ids = ids + ids[: (5 - len(ids))]
        preds.append(" ".join(ids[:5]))
else:
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
        arch = resnet18
        body = create_body(arch, pretrained=True, cut=-2)
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
            test_df_for_dl, bs=bs, shuffle_train=False, num_workers=_NW, pin_memory=_PIN
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

        preds = []
        for i in range(0, test_emb.shape[0], chunk):
            te = test_emb[i : i + chunk]  # [c, d]
            sims = te @ train_emb_small_t  # [c, n_train_small]
            topk_idx = (
                torch.topk(sims, k=min(K, sims.shape[1]), dim=1).indices.cpu().numpy()
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
                        train_meta_small["hotel_id"].value_counts().index.tolist()[:10]
                    )
                    for hid in pad_ids:
                        if len(top_ids) >= 5:
                            break
                        if hid not in top_ids:
                            top_ids.append(hid)
                preds.append(" ".join(top_ids[:5]))

        if len(preds) != len(submission):
            raise RuntimeError(
                f"Predictions length {len(preds)} does not match submission length {len(submission)}"
            )

preds[:3], len(preds)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1616684629.py in <cell line: 0>()
     89         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     90         arch = resnet18
---> 91         body = create_body(arch, pretrained=True, cut=-2)
     92         emb_model = (
     93             nn.Sequential(body, AdaptiveConcatPool2d(), Flatten()).to(device).eval()

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in create_body(model, n_in, pretrained, cut)
     85         ll = list(enumerate(model.children()))
     86         cut = next(i for i,o in reversed(ll) if has_pool_type(o))
---> 87     return cut_model(model, cut)
     88 
     89 # %% ../../nbs/21_vision.learner.ipynb 20

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in cut_model(model, cut)
     74 def cut_model(model, cut):
     75     "Cut an instantiated model"
---> 76     if   isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
     77     elif callable(cut): return cut(model)
     78     raise NameError("cut must be either integer or a function")

AttributeError: 'function' object has no attribute 'children'

## === cell 9
submission_out = submission.copy()
submission_out["hotel_id"] = preds

submission_out["image"] = submission_out["image"].astype(str)
submission_out["hotel_id"] = submission_out["hotel_id"].astype(str)

submission_out.to_csv("submission.csv", index=False)
submission_out.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/579005031.py in <cell line: 0>()
      1 submission_out = submission.copy()
----> 2 submission_out["hotel_id"] = preds
      3 
      4 submission_out["image"] = submission_out["image"].astype(str)
      5 submission_out["hotel_id"] = submission_out["hotel_id"].astype(str)

NameError: name 'preds' is not defined
