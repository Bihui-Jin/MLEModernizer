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

0.00209

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I fix the missing model files by switching to a robust path discovery (searching under `../input` and `/kaggle/input`) and gracefully skipping any models that still can’t be found, so the notebook runs end-to-end. I also fix the ensemble accumulation bug (`probs` was never updated, so later models weren’t actually combined) while preserving the same log-prob averaging approach. To ensure a valid submission is always produced, I add a safe fallback that fills predictions with the most frequent training `hotel_id` values if no models are loadable. Finally, I keep the submission format exactly as required (`image,hotel_id`) and write `submission.csv` to the working directory.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.5399), so we need a real accuracy lift with minimal disruption to your pipeline. The biggest issue is that your code likely isn’t loading the intended pretrained models at all (or is loading mismatched ones), so it falls back to “top5 most frequent” predictions which score near-zero on MAP@5. I keep your exact inference/ensemble logic, but make model discovery robust to this competition’s dataset layout by searching specifically under `../input` for any `.pkl` exports and preferring files that look like hotel-id exports, and I fail loudly (with a controlled message) if we still load zero models so you don’t unknowingly submit a fallback. I also fix the test image path to match the actual provided structure and ensure the test DataLoader is built from file paths that fastai can open reliably.'

# 9. Code solution

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
image_path = (
    (base_input / "test_images")
    if (base_input / "test_images").exists()
    else Path("../input/hotel-id-2021-fgvc8/test_images/")
)
image_path = str(image_path) + ("" if str(image_path).endswith("/") else "/")
image_path



## === cell 5
sample_sub_path = base_input / "sample_submission.csv"
if not sample_sub_path.exists():
    sample_sub_path = Path("../input/hotel-id-2021-fgvc8/sample_submission.csv")

submission = pd.read_csv(sample_sub_path)

test = submission.copy()
test["image"] = (
    (Path(image_path).name + "/" + test["image"])
    if "hotel-id-2021-fgvc8" not in image_path
    else "hotel-id-2021-fgvc8/test_images/" + test["image"]
)
test["image"] = test["image"].apply(
    lambda x: (
        str((Path(image_path).parent / Path(x).name).resolve())
        if (Path(image_path) / Path(x).name).exists()
        else str((Path(image_path) / Path(x).name).resolve())
    )
)
test.head()




## === cell 6
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

        test_dl = learn_tmp.dls.test_dl(test)
        probs_temp, _ = learn_tmp.tta(dl=test_dl)

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

if not loaded_any:
    raise RuntimeError(
        "No exported fastai learner models were loadable. "
        "Refusing to create a fallback submission (it scores near 0). "
        "Please ensure the model .pkl exports are present under ../input or /kaggle/input."
    )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2208623326.py in <cell line: 0>()
     87 # This directly targets score improvement by preventing accidental bad submissions.
     88 if not loaded_any:
---> 89     raise RuntimeError(
     90         "No exported fastai learner models were loadable. "
     91         "Refusing to create a fallback submission (it scores near 0). "

RuntimeError: No exported fastai learner models were loadable. Refusing to create a fallback submission (it scores near 0). Please ensure the model .pkl exports are present under ../input or /kaggle/input.

## === cell 7
if loaded_any and probs2 is not None:
    preds_idx = probs2.topk(5, dim=1)[1].cpu().numpy()
else:
    preds_idx = None

preds_idx is None, (None if preds_idx is None else preds_idx.shape)



## === cell 8
if preds_idx is not None:
    vocab = np.array(learn.dls.vocab)
    preds = [" ".join(map(str, vocab[row])) for row in preds_idx]
else:
    train_path = base_input / "train.csv"
    if not train_path.exists():
        train_path = Path("../input/hotel-id-2021-fgvc8/train.csv")
    train_df = pd.read_csv(train_path)
    top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    fallback_pred = " ".join(top5)
    preds = [fallback_pred] * len(submission)

preds[:3], len(preds)



## === cell 9
submission_out = submission.copy()
submission_out["hotel_id"] = preds

submission_out["image"] = submission_out["image"].astype(str)
submission_out["hotel_id"] = submission_out["hotel_id"].astype(str)

submission_out.to_csv("submission.csv", index=False)
submission_out.head()
