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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.8370631752274111

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the runtime failure by making the weight-loading robust: search common Kaggle input locations for the expected `.pt` files and load them with `map_location` so it works on CPU/GPU. If the external weights are not present, the code still run end-to-end by falling back to a safe baseline prediction derived from `train.csv` label prevalences (score be lower than a real model, but it produce a valid submission instead of crashing). I also fix the submission column mismatch by writing predictions into the exact columns from `sample_submission.csv` (your current sample file has only 9 target columns), ensuring the output CSV is valid. Core model/inference logic is preserved; the only behavioral change is the necessary fallback when checkpoints are missing.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score strongly suggests you are always hitting the “no checkpoints found” fallback (constant priors), which is typically near-random for mean AUC. To move toward the 0.837 target without changing core model logic, I (1) expand checkpoint discovery to specifically search for your `rccl-1x144-f-train` directory and accept common filename variants, and (2) instantiate a fresh model per checkpoint so repeated `load_state_dict` calls can’t silently leave mixed weights if a load partially fails. I also make the output dimension handling robust by reading the true set of competition target columns from `train.csv` and ensuring the submission includes any missing columns (your `sample_submission.csv` appears truncated in this environment), which prevents accidental column-drop that can tank score. If checkpoints still aren’t present, the code still produce a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score indicates you’re still effectively submitting near-constant predictions (either because checkpoints aren’t being found/loaded, or because predictions aren’t aligned to the exact 11 competition targets). I make two minimal, score-relevant fixes: (1) ensure we always load the *correct* 11 target columns and submission schema by deriving columns from `train.csv` (not the possibly-truncated `sample_submission.csv`) and adding any missing columns in the right order, and (2) strengthen checkpoint discovery/loading to handle common checkpoint formats (plain `state_dict`, wrapped dict with `state_dict`, or `model_state_dict`) and common filename/root variants so the real trained model is used rather than the fallback priors. These changes preserve your model and inference logic; they only prevent silent fallback/misalignment that can collapse AUC to ~0.5. The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly indicates the script is still submitting near-constant predictions because no real checkpoints are being loaded; to move toward the 0.837 target we should make checkpoint discovery/load actually succeed without changing the CNN or inference. I (1) extend checkpoint discovery to also look for `.pt/.pth` files anywhere under `../input` and `/kaggle/input` that match your epoch numbers (19/20/23/25), and (2) relax `load_state_dict` to `strict=False` but only as a fallback when strict loading fails, preventing “all-or-nothing” load failures from forcing the priors fallback. I also fix a critical submission-schema issue: your `sample_submission.csv` in this environment is missing some required target columns, so we always write a full 11-label submission in the correct order (as per `train.csv`), which avoids implicit scoring collapse. These are minimal changes that preserve the core model/inference and only address weight-loading + submission correctness.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is consistent with the “no checkpoints loaded → constant priors” fallback, which produces near-random mean AUC. To move toward the 0.837 target without changing your CNN/inference logic, I (1) fix the critical output-dimension mismatch by setting the model output layer to the true number of targets (derived from `train.csv`, i.e., 11), avoiding an always-missing-column situation; and (2) make checkpoint discovery more likely to succeed by also scanning the competition dataset directory itself (`BASE_INPUT`) in addition to `../input` and `/kaggle/input`. I also ensure the submission columns exactly match the 11 training targets in the correct order (not the truncated sample file), which prevents schema-related score collapse. These are minimal, score-relevant changes; the model architecture and inference remain the same aside from correcting the final layer size to match the task labels.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is consistent with the current script still submitting almost-constant predictions because the code never actually finds/loads real checkpoints (so it falls back to train priors), and also because the model output layer is fixed to 11 while your discovered checkpoints may have been trained with 9 outputs (matching your truncated sample submission), causing strict loading to fail and silently keep random weights. I make two minimal, score-relevant fixes: (1) set the model output dimension from the checkpoint itself when a checkpoint is found (so weight loading succeeds), and then map/pad predictions into the full 11 competition targets; and (2) strengthen checkpoint discovery to explicitly include the current working directory and the competition dataset folder so your `.pt/.pth` files are actually located if present. This preserves your CNN, inference loop, and sigmoid post-processing; it only makes weight loading and output-dimension alignment robust so you stop hitting the constant-prior fallback and move the score upward toward the target. The script still always write a valid `./submission.csv` with all 11 required target columns in `train.csv` order.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests the model checkpoints still aren’t being used (so you’re effectively submitting constant priors), or that weights load but BatchNorm is behaving inconsistently at inference. I make two minimal, score-relevant changes: (1) broaden checkpoint discovery to also include the competition dataset folder itself (`BASE_INPUT`) so `Epoch_*.pt` can actually be found in this environment, and (2) load checkpoints once (not twice) and force deterministic inference by running a one-time BN “calibration” pass on a small subset of training images before predicting test (this doesn’t change architecture or training, it just aligns BN running stats to the data distribution, often materially improving AUC). The submission still be written with all 11 true target columns from `train.csv` (even if the provided `sample_submission.csv` is truncated), ensuring a valid file schema.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is consistent with the current script still producing near-constant predictions because it likely never finds/loads real checkpoints in this environment; to move toward the 0.837 target we should make checkpoint discovery succeed without changing the CNN or inference loop. I make a minimal, score-relevant fix by explicitly scanning for any `.pt/.pth` files and selecting only those whose state_dict actually matches your model keys (so we don’t “load” incompatible checkpoints and end up with random outputs). I also fix the padding bug where `priors_full[ckpt_ol:]` is the wrong slice when mapping checkpoint outputs to the full 11 labels; instead we pad missing columns with the corresponding tail of `priors_full` (based on `n_targets - ckpt_ol`). Finally, I ensure the submission schema is exactly the 11 true targets from `train.csv` in the correct order, regardless of the truncated `sample_submission.csv` in this environment.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is consistent with predictions not being evaluated correctly (missing required target columns) and/or still not actually using model checkpoints (falling back to constant priors). To move toward the 0.837 target with minimal change, I (1) force the submission schema to include the full 11 label columns in the exact order from `train.csv` (your `sample_submission.csv` here is truncated to 9 labels, which can collapse the score), and (2) make checkpoint loading more likely to succeed by selecting the best-matching checkpoint(s) by key-overlap and using strict loading when possible (only falling back to `strict=False` when necessary). I keep your CNN, preprocessing, inference, and sigmoid post-processing intact, and still guarantee `./submission.csv` is always written end-to-end.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is consistent with still submitting near-constant predictions because the script likely isn’t finding/using any compatible checkpoints, and because it currently reads the test IDs from a truncated `sample_submission.csv` (only 3009 rows here) rather than from the actual test directory. I make two minimal, score-relevant fixes: (1) build the test ID list by scanning `TEST_DIR` so the submission has the correct full row count and aligns to the evaluation set, and (2) improve checkpoint selection by (a) filtering for checkpoints that both match keys and have a compatible output layer size, and (b) allowing a slightly lower key-overlap threshold when strict loading succeeds—this increases the chance we actually use trained weights rather than falling back to priors. Core model/inference logic is unchanged: same CNN, same sigmoid, same dataloading approach; these changes only prevent schema/alignment and checkpoint-discovery issues that can collapse AUC to ~0.5. The script still always writes `./submission.csv` with all 11 target columns in `train.csv` order.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is still consistent with “valid CSV but effectively random/constant predictions”, and in this script the biggest remaining risk is that the test set is being read incorrectly (your sample_submission is truncated to 3009 IDs, and your directory appears nested like `.../test/test`). I make a minimal, score-relevant fix to robustly locate the real test image directory (including the common nested `test/test` case) and always build `StudyInstanceUID` from the actual `.jpg` files, which should immediately move AUC upward if you were previously predicting for the wrong/partial ID set. I also ensure we always load test images from that resolved directory and keep submission columns exactly the 11 targets from `train.csv` (still writing a valid `submission.csv`). Core model, inference, and checkpoint logic remain unchanged.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is consistent with either (a) still falling back to constant priors (no compatible checkpoints actually loaded), or (b) a submission schema mismatch: in your environment `sample_submission.csv` is truncated to 9 targets, while the competition requires 11, which can tank the score. I make two minimal, score-relevant fixes: (1) always build the submission schema from `train.csv`’s 11 target columns and then reindex to the *exact* competition column order (so no missing/extra columns), and (2) make checkpoint selection more likely to load real weights by also accepting high-overlap checkpoints even if overlap is slightly below 0.65 (but still requiring strict load to succeed), and ensuring the model output dim matches the checkpoint’s FC3 exactly when loading. Core CNN/inference (same preprocessing, same forward, same sigmoid, same averaging) is preserved; we’re just preventing silent fallback/misalignment that yields ~0.5. The script still always write a valid `./submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is still most consistent with (a) predicting for the wrong/partial test set (ID misalignment/row count mismatch), and/or (b) still not actually loading usable checkpoints so you fall back to constant priors. I make two minimal, score-relevant fixes: robustly resolve the actual `train/` and `test/` image directories (handling the common nested `train/train` and `test/test` folders) and always build the submission ID list from the resolved test image directory. I also add a hard guard that enforces the submission row count matches the number of discovered test images (and only falls back to sample_submission IDs if no images are found), which prevents silent misalignment that can collapse AUC. Core model, preprocessing, inference, sigmoid, and checkpoint logic remain the same; these changes only ensure we’re predicting for the correct evaluation set and reading images from the correct paths.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is still most consistent with a schema/row alignment problem rather than the CNN itself: in this environment `sample_submission.csv` is truncated (missing 2 target columns), and if the written CSV doesn’t exactly match Kaggle’s required 12 columns (UID + 11 targets) Kaggle effectively treats missing columns as constant/incorrect, collapsing mean AUC toward ~0.5. I make the smallest score-relevant change: always build the submission template from `train.csv`’s 11 targets, fill it with your predictions, and then write exactly those columns (in correct order) while keeping your model/inference/checkpoint logic intact. I also add a hard guard that ensures we never silently write fewer than 11 targets even if the local sample file is truncated, and keep the test UID discovery from the actual test image directory. These changes should move the score upward toward your 0.837 target without altering the core model.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is still most consistent with a schema mismatch: in this environment the provided `sample_submission.csv` is truncated (9 targets), and your current script writes 11 targets; if Kaggle expects 11 but your local sample is truncated, we should still keep 11 (from `train.csv`)—however, the bigger risk is that your `target_cols` order may not match Kaggle’s expected canonical order, which can silently crater mean AUC even with good probabilities. I make a minimal, score-relevant change to enforce the official competition column order (hard-coded from the problem statement) and then reindex both predictions and priors to that order. I also add a small guard that prevents accidentally training/predicting with a different label ordering when checkpoints have 9/10 outputs by mapping/padding into the canonical 11 in the correct positions (not just “append priors at the end”). Core CNN/inference/checkpoint logic remains the same; this only fixes label/column alignment so your score can move upward toward 0.837.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
import torch.nn.functional as F

import gc
import os
import cv2
from time import time

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



## === cell 1
BASE_INPUT = "../input/ranzcr-clip-catheter-line-classification"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test")

ALT_BASE_INPUT = "/kaggle/input/ranzcr-clip-catheter-line-classification"
if not os.path.exists(SAMPLE_SUB) and os.path.exists(ALT_BASE_INPUT):
    BASE_INPUT = ALT_BASE_INPUT
    TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
    SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
    TEST_DIR = os.path.join(BASE_INPUT, "test")




## === cell 2
def breaker():
    print("\n" + 50 * "-" + "\n")


def head(x, no_of_ele=5):
    print(x[:no_of_ele])


def getImages(file_path=None, file_names=None, size=None):
    images = []
    for name in file_names:
        try:
            image = cv2.imread(
                os.path.join(file_path, name + ".jpg"), cv2.IMREAD_GRAYSCALE
            ).astype("float64")
        except AttributeError:
            print(os.path.join(file_path, name + ".jpg"))
            image = np.zeros((size, size), dtype="float64")
        if size:
            image = cv2.resize(
                image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4
            )
        cv2.normalize(src=image, dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        images.append(image.reshape(1, size, size))
    return np.array(images)




## === cell 3
def find_existing_checkpoint(rel_path, extra_search_dirs=None):
    """
    Score-relevant: broaden checkpoint discovery so we load trained weights
    instead of falling back to constant priors (which tends to score ~0.5 AUC).
    """
    if extra_search_dirs is None:
        extra_search_dirs = []

    candidates = []
    if rel_path:
        candidates.extend(
            [
                rel_path,
                os.path.join(
                    "../input",
                    os.path.basename(os.path.dirname(rel_path)),
                    os.path.basename(rel_path),
                ),
                os.path.join(
                    "/kaggle/input",
                    os.path.basename(os.path.dirname(rel_path)),
                    os.path.basename(rel_path),
                ),
            ]
        )

    if rel_path:
        base = os.path.basename(rel_path)
        candidates.extend(
            [
                os.path.join("../input/rccl-1x144-f-train", base),
                os.path.join("/kaggle/input/rccl-1x144-f-train", base),
                os.path.join("../input/rccl-1x144-f-train/rccl-1x144-f-train", base),
                os.path.join(
                    "/kaggle/input/rccl-1x144-f-train/rccl-1x144-f-train", base
                ),
            ]
        )

    if rel_path:
        base = os.path.basename(rel_path)
        root, ext = os.path.splitext(base)
        variants = list(
            dict.fromkeys(
                [
                    base,
                    root + ".pt",
                    root + ".pth",
                    root.lower() + ".pt",
                    root.lower() + ".pth",
                ]
            )
        )
        for v in variants:
            candidates.extend(
                [
                    os.path.join("../input/rccl-1x144-f-train", v),
                    os.path.join("/kaggle/input/rccl-1x144-f-train", v),
                    os.path.join("../input/rccl-1x144-f-train/rccl-1x144-f-train", v),
                    os.path.join(
                        "/kaggle/input/rccl-1x144-f-train/rccl-1x144-f-train", v
                    ),
                ]
            )

    if rel_path:
        base = os.path.basename(rel_path)
        candidates.extend(
            [
                os.path.join(".", base),
                os.path.join("./checkpoints", base),
                os.path.join("/kaggle/working", base),
                os.path.join("/kaggle/working/checkpoints", base),
            ]
        )

    if rel_path:
        base = os.path.basename(rel_path)
        for d in extra_search_dirs:
            if d:
                candidates.append(os.path.join(d, base))

    for p in candidates:
        if p and os.path.exists(p):
            return p

    base_dirs = ["../input", "/kaggle/input", ".", "/kaggle/working"] + [
        d for d in extra_search_dirs if d
    ]
    target_name = os.path.basename(rel_path) if rel_path else None

    name_variants = []
    if target_name:
        root, ext = os.path.splitext(target_name)
        name_variants.extend([target_name, target_name.lower()])
        name_variants.extend(
            [root + ".pt", root + ".pth", root.lower() + ".pt", root.lower() + ".pth"]
        )
        name_variants.append(target_name.replace("Epoch_", "epoch_"))
        name_variants.append(target_name.replace("Epoch_", "Epoch"))
        name_variants = list(dict.fromkeys(name_variants))

    for bd in base_dirs:
        if os.path.exists(bd) and target_name:
            for rootdir, dirs, files in os.walk(bd):
                for nv in name_variants:
                    if nv in files:
                        return os.path.join(rootdir, nv)
    return None


def find_checkpoint_by_epoch(epoch_num, exts=(".pt", ".pth"), extra_search_dirs=None):
    """
    Score-relevant: locate any checkpoint containing 'Epoch_{n}' under common dirs.
    """
    if extra_search_dirs is None:
        extra_search_dirs = []

    patterns = [
        f"Epoch_{epoch_num}",
        f"epoch_{epoch_num}",
        f"epoch{epoch_num}",
        f"Epoch{epoch_num}",
    ]
    for bd in ["../input", "/kaggle/input", ".", "/kaggle/working"] + [
        d for d in extra_search_dirs if d
    ]:
        if not os.path.exists(bd):
            continue
        for rootdir, dirs, files in os.walk(bd):
            for fn in files:
                lower = fn.lower()
                if not lower.endswith(exts):
                    continue
                if any(p.lower() in lower for p in patterns):
                    return os.path.join(rootdir, fn)
    return None


def find_all_checkpoints(exts=(".pt", ".pth"), extra_search_dirs=None, limit=200):
    """
    Score-relevant: scan for candidate .pt/.pth; later we'll filter by compatibility/overlap.
    """
    if extra_search_dirs is None:
        extra_search_dirs = []
    base_dirs = ["../input", "/kaggle/input", ".", "/kaggle/working"] + [
        d for d in extra_search_dirs if d
    ]
    found = []
    for bd in base_dirs:
        if not os.path.exists(bd):
            continue
        for rootdir, dirs, files in os.walk(bd):
            for fn in files:
                if fn.lower().endswith(exts):
                    found.append(os.path.join(rootdir, fn))
                    if len(found) >= limit:
                        return found
    return found




## === cell 4
def get_train_priors(train_csv_path, target_cols):
    """
    Fallback baseline: use label prevalence to fill probabilities if checkpoints are missing.
    Ensures a valid submission is produced end-to-end.
    """
    tr = pd.read_csv(train_csv_path)
    priors = tr[target_cols].mean(axis=0).astype("float64").values
    priors = np.clip(priors, 1e-6, 1 - 1e-6)
    return priors




## === cell 5
start_time = time()

ss = pd.read_csv(SAMPLE_SUB)


def _resolve_image_dir(dir_path, nested_name):
    """
    Score-relevant: the dataset can be nested as .../test/test and .../train/train.
    If we read the wrong folder, we get too few/no images and predictions misalign -> score collapse.
    """
    if dir_path and os.path.isdir(dir_path):
        try:
            if any(fn.lower().endswith(".jpg") for fn in os.listdir(dir_path)):
                return dir_path
        except Exception:
            pass
        nested = os.path.join(dir_path, nested_name)
        if os.path.isdir(nested):
            try:
                if any(fn.lower().endswith(".jpg") for fn in os.listdir(nested)):
                    return nested
            except Exception:
                pass
    return dir_path


def list_uids_from_dir(img_dir):
    uids = []
    if not img_dir or (not os.path.exists(img_dir)):
        return uids
    for fn in os.listdir(img_dir):
        if fn.lower().endswith(".jpg"):
            uids.append(os.path.splitext(fn)[0])
    uids.sort()
    return np.array(uids, dtype=object)


TEST_DIR = _resolve_image_dir(TEST_DIR, "test")
TRAIN_DIR = _resolve_image_dir(os.path.join(BASE_INPUT, "train"), "train")

print("Resolved TEST_DIR:", TEST_DIR)
print("Resolved TRAIN_DIR:", TRAIN_DIR)

ts_img_names = list_uids_from_dir(TEST_DIR)
print("Found test images:", len(ts_img_names))

if len(ts_img_names) == 0:
    ts_img_names = ss["StudyInstanceUID"].values
    print(
        "WARNING: No test images found; falling back to sample_submission IDs:",
        len(ts_img_names),
    )

ts_images = getImages(os.path.join(TEST_DIR, ""), ts_img_names, size=144)

breaker()
print("Time Taken to read data : {:.2f} minutes".format((time() - start_time) / 60))
print("Test images loaded:", ts_images.shape[0])
breaker()



## === cell 6
train_df = pd.read_csv(TRAIN_CSV)

CANONICAL_TARGETS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]

train_targets = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]
missing = [c for c in CANONICAL_TARGETS if c not in train_targets]
extra = [c for c in train_targets if c not in CANONICAL_TARGETS]
if len(missing) > 0:
    raise ValueError(f"train.csv is missing expected target columns: {missing}")
if len(extra) > 0:
    print(
        "WARNING: train.csv has extra non-canonical target columns; they will be ignored:",
        extra,
    )

target_cols = CANONICAL_TARGETS
n_targets = len(target_cols)
print("Using canonical target columns:", n_targets)
print(target_cols)

for c in target_cols:
    if c not in ss.columns:
        ss[c] = 0.0




## === cell 7
class Dataset(Dataset):
    def __init__(this, X=None, y=None, mode="train"):
        this.mode = mode
        this.X = X
        if mode == "train":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        if this.mode == "train":
            return torch.FloatTensor(this.X[idx]), torch.FloatTensor(this.y[idx])
        else:
            return torch.FloatTensor(this.X[idx])




## === cell 8
ORIG_OL = n_targets




## === cell 9
def slice_or_pad_predictions(pred, out_dim, pad_values=None):
    """
    Align prediction dimension to required targets by slicing/padding.
    Padding uses train priors (if provided) rather than zeros to avoid unnecessary score loss.
    """
    if pred.shape[1] == out_dim:
        return pred
    if pred.shape[1] > out_dim:
        return pred[:, :out_dim]
    if pad_values is None:
        pad_values = np.zeros((out_dim - pred.shape[1],), dtype=pred.dtype)
    pad_values = pad_values.astype(pred.dtype)
    pad = np.tile(pad_values.reshape(1, -1), (pred.shape[0], 1))
    return np.concatenate([pred, pad], axis=1)




## === cell 10
class CFG:
    tr_batch_size = 128  # Also va_batch_size
    ts_batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1
    OL = ORIG_OL

    def __init__(
        this, filter_sizes=[64, 128, 256, 512], HL=[4096, 4096], epochs=50, n_folds=5
    ):
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds




## === cell 11
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[4096, 4096], epochs=50, n_folds=5)




## === cell 12
class CNN(nn.Module):
    def __init__(
        this,
        in_channels=1,
        filter_sizes=None,
        HL=None,
        OL=None,
        use_DP=False,
        DP1=0.2,
        DP2=0.5,
    ):
        super(CNN, this).__init__()

        this.use_DP = use_DP

        this.DP1 = nn.Dropout(p=0.2)
        this.DP2 = nn.Dropout(p=0.5)

        this.MP_ = nn.MaxPool2d(kernel_size=2)

        this.CN1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filter_sizes[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        this.CN2 = nn.Conv2d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        this.CN3 = nn.Conv2d(
            in_channels=filter_sizes[1],
            out_channels=filter_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        this.CN4 = nn.Conv2d(
            in_channels=filter_sizes[2],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN5 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN6 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN6 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.FC1 = nn.Linear(in_features=filter_sizes[3] * 2 * 2, out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=HL[1])
        this.FC3 = nn.Linear(in_features=HL[1], out_features=OL)

    def getOptimizer(this, A_S=True, lr=1e-3, wd=0):
        if A_S:
            return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.SGD(this.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def getStepLR(this, optimizer=None, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma
        )

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma
        )

    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-6):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=patience, eps=eps, verbose=True
        )

    def forward(this, x):
        if not this.use_DP:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.FC1(x))
            x = F.relu(this.FC2(x))
            x = this.FC3(x)

            return x
        else:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.DP2(this.FC1(x)))
            x = F.relu(this.DP2(this.FC2(x)))
            x = this.FC3(x)

            return x




## === cell 13
def _extract_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state_dict = state["state_dict"]
    elif (
        isinstance(state, dict)
        and "model_state_dict" in state
        and isinstance(state["model_state_dict"], dict)
    ):
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    if isinstance(state_dict, dict):
        keys = list(state_dict.keys())
        if len(keys) > 0 and keys[0].startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_output_dim_from_ckpt(ckpt_path, device):
    """
    Infer OL from FC3 weights in the checkpoint to avoid strict-load failures and random weights.
    """
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return None
    state = torch.load(ckpt_path, map_location=device)
    sd = _extract_state_dict(state)
    if not isinstance(sd, dict):
        return None
    if "FC3.weight" in sd and hasattr(sd["FC3.weight"], "shape"):
        return int(sd["FC3.weight"].shape[0])
    return None


def safe_load_state_dict(model, ckpt_path, device):
    """
    Robustly load checkpoint; try strict=True then strict=False.
    """
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return False
    state = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(state)

    try:
        model.load_state_dict(state_dict, strict=True)
        return True
    except RuntimeError:
        try:
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False


def checkpoint_key_overlap(ckpt_path, model, device):
    """
    Score-relevant: measure compatibility to prioritize correct checkpoints and avoid random weights.
    """
    try:
        state = torch.load(ckpt_path, map_location=device)
        sd = _extract_state_dict(state)
        if not isinstance(sd, dict):
            return 0.0
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(sd.keys())
        return float(len(model_keys & ckpt_keys) / max(1, len(model_keys)))
    except Exception:
        return 0.0




## === cell 14
def bn_calibrate(model, dataloader, device, max_batches=16):
    """
    BN calibration (inference-only) can materially improve AUC without changing architecture/training.
    Kept as-is; small max_batches to stay within runtime.
    """
    model.to(device)
    model.train()
    seen = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            _ = model(X)
            seen += 1
            if seen >= max_batches:
                break
    model.eval()
    return model


def predict_(model=None, dataloader=None, device=None):
    model.to(device)
    model.eval()

    y_pred = torch.zeros(0, model.FC3.out_features, device=device)
    for X in dataloader:
        X = X.to(device)
        with torch.no_grad():
            Pred = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, Pred), dim=0)

    return y_pred.detach().cpu().numpy()




## === cell 15
ts_data_setup = Dataset(ts_images, None, "test")
ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

tr_small = train_df[["StudyInstanceUID"] + target_cols].copy()
tr_small = tr_small.sample(n=min(2048, len(tr_small)), random_state=seed).reset_index(
    drop=True
)
tr_img_names = tr_small["StudyInstanceUID"].values

tr_images_small = getImages(os.path.join(TRAIN_DIR, ""), tr_img_names, size=144)
tr_labels_small = tr_small[target_cols].values.astype("float32")
tr_data_setup = Dataset(tr_images_small, tr_labels_small, "train")
tr_data_small = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=False)

priors_full = get_train_priors(TRAIN_CSV, target_cols)

extra_dirs = [
    BASE_INPUT,
    os.path.join(BASE_INPUT, "ranzcr-clip-catheter-line-classification"),
    ".",
    "/kaggle/working",
]

paths = [
    "../input/rccl-1x144-f-train/Epoch_19.pt",
    "../input/rccl-1x144-f-train/Epoch_20.pt",
    "../input/rccl-1x144-f-train/Epoch_23.pt",
    "../input/rccl-1x144-f-train/Epoch_25.pt",
]

epoch_nums = [19, 20, 23, 25]
for e in epoch_nums:
    p2 = find_checkpoint_by_epoch(e, extra_search_dirs=extra_dirs)
    if p2 is not None and p2 not in paths:
        paths.append(p2)

all_ckpts = find_all_checkpoints(extra_search_dirs=extra_dirs, limit=200)
for p in all_ckpts:
    if p not in paths:
        paths.append(p)

resolved = []
seen = set()
for p in paths:
    ckpt_found = (
        find_existing_checkpoint(p, extra_search_dirs=extra_dirs) if p else None
    )
    if ckpt_found is None or (not os.path.exists(ckpt_found)):
        continue
    if ckpt_found in seen:
        continue
    seen.add(ckpt_found)

    ckpt_ol = infer_output_dim_from_ckpt(ckpt_found, cfg.device)
    if ckpt_ol is None:
        ckpt_ol = cfg.OL

    if ckpt_ol not in (n_targets, 9, 10, 11):
        continue

    probe_model = CNN(
        in_channels=cfg.in_channels,
        filter_sizes=cfg.filter_sizes,
        HL=cfg.HL,
        OL=ckpt_ol,
    )
    ov = checkpoint_key_overlap(ckpt_found, probe_model, cfg.device)
    resolved.append((ov, ckpt_found, ckpt_ol))

resolved.sort(key=lambda x: x[0], reverse=True)
print("Top checkpoint candidates by key-overlap (first 10):")
print(resolved[:10])

preds = []
loaded_any = False
used_ckpts = []

min_overlap = 0.50


def map_ckpt_pred_to_full(pred_ckpt, ckpt_ol, priors_full, full_dim=11):
    out = np.tile(priors_full.reshape(1, -1), (pred_ckpt.shape[0], 1)).astype(
        pred_ckpt.dtype
    )
    use_k = min(ckpt_ol, full_dim)
    out[:, :use_k] = pred_ckpt[:, :use_k]
    return out


for ov, ckpt_found, ckpt_ol in resolved:
    if ov < min_overlap:
        continue

    model = CNN(
        in_channels=cfg.in_channels,
        filter_sizes=cfg.filter_sizes,
        HL=cfg.HL,
        OL=ckpt_ol,
    )

    ok = safe_load_state_dict(model, ckpt_found, cfg.device)
    if not ok:
        continue

    model = bn_calibrate(model, tr_data_small, cfg.device, max_batches=16)
    pred_ckpt = predict_(model=model, dataloader=ts_data, device=cfg.device)

    loaded_any = True

    if pred_ckpt.shape[1] != ckpt_ol:
        pred_ckpt = slice_or_pad_predictions(pred_ckpt, ckpt_ol, pad_values=None)

    pred_full = map_ckpt_pred_to_full(
        pred_ckpt, ckpt_ol, priors_full, full_dim=n_targets
    )
    pred_full = np.clip(pred_full, 1e-15, 1 - 1e-15)
    preds.append(pred_full)
    used_ckpts.append((ckpt_found, ckpt_ol, ov))
    print(f"Loaded checkpoint: {ckpt_found} (ckpt_ol={ckpt_ol}, overlap={ov:.3f})")

    if len(preds) >= 4:
        break

if loaded_any:
    y_pred = np.mean(preds, axis=0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
else:
    y_pred = np.tile(priors_full.reshape(1, -1), (len(ts_img_names), 1))

sub = pd.DataFrame({"StudyInstanceUID": ts_img_names})
for i, c in enumerate(target_cols):
    sub[c] = y_pred[:, i].astype("float64")

assert sub.shape[1] == 1 + n_targets, "Submission does not contain all target columns."
sub = sub.reindex(columns=["StudyInstanceUID"] + target_cols)

assert sub.shape[0] == len(
    ts_img_names
), "Row count mismatch between IDs and predictions."
assert list(sub.columns) == (
    ["StudyInstanceUID"] + target_cols
), "Column order mismatch."

sub.to_csv("./submission.csv", index=False)

print(sub.head(5))
print("Saved submission to ./submission.csv with shape:", sub.shape)
print("Loaded any checkpoints:", loaded_any)
if loaded_any:
    print("Ensembled checkpoints used:", len(preds))
    print("Used checkpoints:", used_ckpts)
else:
    print("Used priors fallback only.")
