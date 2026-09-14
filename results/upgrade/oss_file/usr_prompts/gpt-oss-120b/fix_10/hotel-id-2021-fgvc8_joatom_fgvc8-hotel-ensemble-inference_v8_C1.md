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

0.616496169066261

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00092) has done: 'The fix completes the unfinished conditional, correctly builds the top‑5 prediction strings, and ensures a CSV file named `submission.csv` is written. Minor adjustments include converting image paths to strings, handling the case where no model loads (fallback to the most common hotel), and using `torch.topk` on the accumulated probability tensor. The core model‑loading and ensembling logic remains unchanged.'
- What this solution (achieved 0.00209) has done: 'I adjust the inference cell so that when no model files are found (or they all fail to load) the fallback prediction uses the five most frequent hotels in the training data instead of repeating a single ID, which should raise MAP@5. Additionally, when any models do load I average the accumulated probabilities across models before extracting the top‑5 predictions, a small change that keeps the original ensemble logic while improving calibration. These tweaks keep the core workflow intact and aim to move the score much closer to the target.'
- What this solution (achieved 0.00209) has done: 'I adjust the model file paths to the proper absolute location under `/kaggle/input/…` so the pre‑trained learners can be loaded, average the probabilities whenever at least one model is loaded, and keep the fallback to the five most frequent hotels unchanged. These minimal fixes let the original ensemble logic run and should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I add a simple, data‑driven fallback: when none of the pre‑trained models can be loaded, the script predict the five most frequent hotels *per chain* (derived from the test image’s parent folder) instead of a single global list. This keeps the original workflow intact, only improves the “no‑model” case, and should raise MAP@5 toward the target without altering the core model logic.'
- What this solution (achieved 0.00209) has done: 'I keep the original workflow intact and only adjust the fallback prediction logic. When no pretrained model can be loaded, the code now builds a combined top‑5 list: it starts with the five most frequent hotels for the image’s chain (if known) and fills any remaining slots with the global most‑frequent hotels, avoiding duplicates. This modest change increases the chance that the correct hotel appears in the submitted list, moving the MAP@5 score closer to the target while preserving all core logic.'
- What this solution (achieved 0.00209) has done: 'I add a small discovery step that automatically finds any exported FastAI‑learner `.pkl` files under the `/kaggle/input` directory, so the script can actually load the pretrained models that are present in the environment instead of relying on hard‑coded (non‑existent) paths. This modest change lets the original ensemble logic run when models are found, which should raise MAP@5 toward the target while keeping all core logic unchanged. If no models are discovered the existing fallback (chain‑wise top‑5 plus global top‑5) remains unchanged.'
- What this solution (achieved 0.00209) has done: 'I expanded the model discovery to look for **any** FastAI learner `.pkl` files under both `/kaggle/input` and `/kaggle/working` (instead of only `export_*.pkl`). This lets the script actually load pretrained models that may be present, while keeping the original ensemble / fallback workflow unchanged, so the MAP@5 score should move closer to the target.'
- What this solution (achieved 0.00209) has done: 'I adjust the inference loop so that the test dataloader receives the list of image file paths (instead of the whole DataFrame) and use `learn.get_preds` for a straightforward probability extraction. This small fix lets any discovered FastAI models actually produce predictions, which should raise the MAP@5 score toward the target while keeping the overall workflow unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import fastai
from fastai.vision.all import *
import dill
import warnings
from pathlib import Path
from collections import Counter




## === cell 1
fastai.__version__, torch.__version__




## === cell 2
model_paths = list(Path("/kaggle/input").rglob("*.pkl")) + list(
    Path("/kaggle/working").rglob("*.pkl")
)
if not model_paths:
    warnings.warn(
        "No *.pkl model files found under /kaggle/input or /kaggle/working; will use fallback predictions."
    )
else:
    print(f"Discovered {len(model_paths)} model file(s):")
    for mp in model_paths:
        print(f"  - {mp}")

image_path = Path("/kaggle/input/hotel-id-2021-fgvc8/test_images/")

submission = pd.read_csv("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
test = submission.copy()
test["image"] = (image_path / test["image"]).astype(str)

train_df = pd.read_csv("/kaggle/input/hotel-id-2021-fgvc8/train.csv")
global_top5 = [str(h) for h, _ in Counter(train_df["hotel_id"]).most_common(5)]

chain_top5 = {}
for chain_val, grp in train_df.groupby("chain"):
    chain_top5[int(chain_val)] = [
        str(h) for h, _ in Counter(grp["hotel_id"]).most_common(5)
    ]

probs = None
learn = None  # keep last loaded learner for vocab reference
model_cnt = 0  # count of successfully evaluated models

for model_path in model_paths:
    p = Path(model_path)
    if not p.exists():
        warnings.warn(f"Model file not found, skipping: {model_path}")
        continue
    try:
        learn = load_learner(fname=p, cpu=True)
        test_dl = learn.dls.test_dl(test["image"])
        probs_temp, _ = learn.get_preds(dl=test_dl)
        if probs is None:
            probs = probs_temp
        else:
            probs += probs_temp
        model_cnt += 1
    except Exception as e:
        warnings.warn(f"Failed to load or evaluate model {model_path}: {e}")

if probs is not None and model_cnt >= 1:
    probs = probs / model_cnt  # average probabilities across loaded models

if probs is None:
    preds = []
    for img_path in test["image"]:
        p = Path(img_path)
        try:
            chain_id = int(p.parent.name)
        except Exception:
            chain_id = None
        if chain_id is not None and chain_id in chain_top5:
            top5 = chain_top5[chain_id].copy()
            for gid in global_top5:
                if len(top5) >= 5:
                    break
                if gid not in top5:
                    top5.append(gid)
        else:
            top5 = global_top5
        preds.append(" ".join(top5))
else:
    top5_idxs = probs.topk(5, dim=1)[1]  # indices tensor
    preds = []
    for idx_row in top5_idxs:
        hotel_ids = [learn.dls.vocab[int(i)] for i in idx_row]
        preds.append(" ".join(map(str, hotel_ids)))




## === cell 3
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)
submission.head()
